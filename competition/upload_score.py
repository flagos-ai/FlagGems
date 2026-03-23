#!/usr/bin/env python3

import argparse
import hashlib
import json
import logging
import os
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# score_details 允许的维度 key 及其满分
SCORE_DIMENSIONS = {
    "functional_correctness": 30,  # 功能正确性
    "performance": 20,  # 性能竞争力
    "test_coverage": 20,  # 测试用例完备性
    "adaptability": 10,  # 开源适配性
    "compatibility": 10,  # 跨平台兼容性
    "readability": 10,  # 代码可读性
}


def generate_signature(params: dict, api_secret: str) -> str:
    """生成请求签名 (与 evaluation_client.py 相同算法)

    规则：只对标量字段签名，跳过 dict/list 类型字段。
    签名覆盖身份和操作字段，嵌套详情数据由 API Key + HTTPS 保证安全。
    """
    sorted_keys = sorted(params.keys())
    parts = []
    for k in sorted_keys:
        v = params[k]
        if isinstance(v, (dict, list)):
            continue
        parts.append(f"{k}={v}")
    sign_string = "&".join(parts)
    sign_string = f"{sign_string}&api_secret={api_secret}"
    return hashlib.sha256(sign_string.encode()).hexdigest()


def validate_score_details(score_details: dict) -> str:
    """校验 score_details 各维度，返回错误信息（空字符串表示通过）"""
    for key, value in score_details.items():
        if key not in SCORE_DIMENSIONS:
            return f"未知维度 '{key}'，允许的维度: {', '.join(SCORE_DIMENSIONS.keys())}"
        if not isinstance(value, (int, float)):
            return f"维度 '{key}' 的值必须是数字，当前值: {value}"
        if value < 0 or value > SCORE_DIMENSIONS[key]:
            return f"维度 '{key}' 的值超出范围 [0, {SCORE_DIMENSIONS[key]}]，当前值: {value}"
    return ""


def upload_score(
    api_base_url: str,
    api_key: str,
    api_secret: str,
    name: str,
    github_id: str,
    github_pr: str,
    score: float,
    note: str = None,
    score_details: dict = None,
) -> bool:
    """
    上传评分到 FlagOS 平台

    Args:
        api_base_url: FlagOS API 基础 URL
        api_key: 评测 API Key
        api_secret: 评测 API Secret
        name: 算子名称（需与平台注册名称完全一致）
        github_id: 选手 GitHub ID
        github_pr: GitHub PR 链接
        score: 总分 (0-100)
        note: 备注信息（可选）
        score_details: 分项得分（可选），格式:
            {"functional_correctness": 27, "performance": 18, ...}

    Returns:
        bool: 是否成功
    """
    url = api_base_url.rstrip("/") + "/api/v1/evaluation/track1-score"
    timestamp = str(int(time.time()))

    # 构建请求体
    data = {
        "name": name,
        "github_id": github_id,
        "github_pr": github_pr,
        "score": score,
    }
    if note:
        data["note"] = note
    if score_details:
        data["score_details"] = score_details

    # 计算签名 (score_details 是 dict，会被签名函数跳过，不影响签名)
    sign_params = {**data, "timestamp": timestamp}
    signature = generate_signature(sign_params, api_secret)

    headers = {
        "Content-Type": "application/json",
        "x-eval-api-key": api_key,
        "x-eval-timestamp": timestamp,
        "x-eval-signature": signature,
    }

    try:
        details_str = f", score_details={score_details}" if score_details else ""
        logger.info(
            f"上传评分: name={name}, github_id={github_id}, score={score}{details_str}"
        )
        response = requests.post(url, headers=headers, json=data, timeout=30)
        result = response.json()

        if response.status_code == 200 and result.get("code") == 200:
            logger.info(f"上传成功: {result.get('message')}")
            return True
        else:
            logger.error(f"上传失败: {result}")
            return False

    except requests.RequestException as e:
        logger.error(f"请求异常: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="赛道一 CI/CD 评分上传",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
分项得分维度及满分:
  functional_correctness  功能正确性   满分 30
  performance             性能竞争力   满分 20
  test_coverage           测试用例完备性 满分 20
  adaptability            开源适配性   满分 10
  compatibility           跨平台兼容性  满分 10
  readability             代码可读性   满分 10

示例:
  --score-details '{"functional_correctness": 27, "performance": 18,
                    "test_coverage": 16, "adaptability": 9,
                    "compatibility": 8, "readability": 10}'
""",
    )
    parser.add_argument("--api-url", help="FlagOS API URL", default=None)
    parser.add_argument("--api-key", help="评测 API Key", default=None)
    parser.add_argument("--api-secret", help="评测 API Secret", default=None)
    parser.add_argument("--name", required=True, help="算子名称（需与平台注册名称完全一致，区分大小写）")
    parser.add_argument("--github-id", required=True, help="选手 GitHub 用户名")
    parser.add_argument("--github-pr", required=True, help="GitHub PR 链接")
    parser.add_argument("--score", required=True, type=float, help="总分 (0-100)")
    parser.add_argument("--note", default=None, help="备注信息")
    parser.add_argument(
        "--score-details",
        default=None,
        help='分项得分 JSON，如 \'{"functional_correctness": 27, "performance": 18, ...}\'',
    )

    args = parser.parse_args()

    # 优先使用命令行参数，其次使用环境变量
    api_url = args.api_url or os.environ.get("EVAL_API_BASE_URL", "")
    api_key = args.api_key or os.environ.get("EVAL_API_KEY", "")
    api_secret = args.api_secret or os.environ.get("EVAL_API_SECRET", "")

    if not all([api_url, api_key, api_secret]):
        logger.error("缺少必要参数: api-url, api-key, api-secret (可通过命令行或环境变量提供)")
        sys.exit(1)

    # 解析 score_details
    score_details = None
    if args.score_details:
        try:
            score_details = json.loads(args.score_details)
        except json.JSONDecodeError as e:
            logger.error(f"--score-details JSON 格式错误: {e}")
            sys.exit(1)

        error = validate_score_details(score_details)
        if error:
            logger.error(f"--score-details 校验失败: {error}")
            sys.exit(1)

    success = upload_score(
        api_base_url=api_url,
        api_key=api_key,
        api_secret=api_secret,
        name=args.name,
        github_id=args.github_id,
        github_pr=args.github_pr,
        score=args.score,
        note=args.note,
        score_details=score_details,
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
