# 已验证的 benchmark skip

## 2026-10-24 完整遍历增量

使用 Gems@4184f11bc6fe0775c3042a9400867e01aa469b8a 的 OOM 后继续收集逻辑，对 421 项进行完整 reference-only 复审，无 NOT_RUN。新增 1,561 条精确 OOM 规则：昇腾 106、海光 392、沐曦 376、摩尔 130、平头哥 109、燧原 445、天数 3。全部为明确 OutOfMemoryError，恢复成功且 skip_eligible=true；不根据未知错误或未执行 case 推断 OOM。

累计 2,352 条规则（2,348 条 OOM、4 条 BF16 不支持）。原始证据为 oom-continue-review-20261024-0hIuL3/results，逐规则来源和完整输入元数据归档在该实验目录的 oom-skip-evidence.json。回放检查新增规则不影响 8,283 个成功 case 和 421 个已有 skip。本增量仍只覆盖已验证的 benchmark case，不据此跳过 correctness 测例；加入 skip 不代表 reference、注入或优化已经通过，需另行真机复测。

## 2026-10-23 增量

在原 370 条规则上新增 421 条明确设备 OOM 规则，累计 791 条（787 条 OOM、4 条 BF16 不支持）。新规则来自 Gems@318c49df3550eae21c94cc9975b373e22d051f29 的 benchmark-fix-review-20261023-dATv1G 审核原始逐 case 记录，异常类型全部为 OutOfMemoryError，不以模糊 UNKNOWN 分类或未执行的 NOT_RUN 生成规则。平台分布为昇腾 22、海光 84、沐曦 80、摩尔 74、平头哥 53、燧原 105、天数 3。原始报告同时收录于 failed_review_pytests_1037.zip（SHA256 92aa37c51f1f009d8e31ff7491fb01e9c74664bc863950502b70f890e8a730c0）。

仍按 vendor、case_id、dtype/shape/params 摘要精确匹配，在输入分配前触发 pytest.skip；不跳过整个算子，不修改输入网格或 reference，不把同一算子后续未执行 case 当作 OOM。规则仅作用于 benchmark，不扩展到 correctness。首次失败后尚有未执行 case，因此这些规则生效后可能继续发现新的错误，不代表 421 项都能最终通过审核或优化。当前部分 KGS 版本对 Preflight 的 skipped 记录仍有兼容问题，需要独立修复，不能通过删除 skip 或放宽候选调用校验掩盖。

`benchmark/verified_case_skips.json` 保存 2026-10-22 的 95 项 reference-only 复测中确认的 370 个 case：昇腾 350 个 OOM、4 个 put BF16 不支持，海光 8 个 OOM，沐曦 8 个 OOM。原始 Gems 为 47246ae229e1c795f63dc6624a407ace2ef54747，KGS 为 95a5f62b23619c35d1523b53afb0c027cff40a63；原始归档为 full-reference-review-20261022-RCw8L3/results。

每行依次为 vendor、case_id、metadata SHA-256、原因。metadata 是 dtype、shape、params 三字段的 JSON，采用 sort_keys=True、separators=(",", ":") 后计算 SHA-256。只有所有字段匹配才在输入分配前触发 pytest.skip；shape、参数或 dtype 改变会使规则失效。list-cases 仍保留原 case，不删除 workload。规则仅覆盖 benchmark，不推导到 correctness。规则不是通用硬件容量声明，换显存容量或运行环境后应重新验证并删除不再适用的条目。

不捕获运行时 OOM 后伪装成功，不跳过 API/kernel 缺失、形状错误、输出 dtype 不一致、共享内存超限或未知错误。全 SKIP 不等于 reference/preflight 通过。reference、preflight 按 case 记录 SKIP 并继续后续 case。

同时迁移 master@4897cc306 的相关已有策略：EXTRA_INT_DTYPES 服从 support_int64，tsingmicro 的 linalg_svdvals FP32 skip，以及 enflame 的 linalg_ldl_factor FP64 skip。保留 kernelgen-dev 的注入接口及已有测试覆盖。master 通用 foreach bool CPU reference skip 对应的新测试结构在此分支尚不存在，当前 per-operator foreach abs 没有 bool 正常测例，因此不引入整套新测试或无效规则。master 缩小 foreach shape 属于 workload 变更，不作为 skip 同步。

## 验证

对原始 1,908 个 case 回放匹配，恰好 370 个命中，另外 1,538 个不变。昇腾、海光、沐曦在 Gems@64b81d01 / KGS@95a5f62b 完成真实 reference-only 回归：正常 case PASSED、指定坏 case SKIP、仅选坏 case 返回 ALL_SKIP；昇腾额外验证 put BF16。各平台 host 回归 61 passed，归档 verified-skips-final-20261022-AioW9W。随后补充单 case preflight 的 pytest.skip 传播，以保持外部注入探针能够识别 SKIP；最终 host 专项 5 passed，覆盖精确匹配、单 case 传播、多 case 继续以及全 SKIP 不通过 preflight。没有把全部 95 个算子重新跑完，不宣称 74 项都已审核通过。
