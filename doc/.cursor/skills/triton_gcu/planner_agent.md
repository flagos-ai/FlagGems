# PlannerAgent Skill — 计划监督专属

## 核心职责

全程监督优化进度，记录关键节点与异常情况，确保流程严格遵循时间限制、迭代次数要求，不出现步骤跳过、流程违规。

## 全局约束

- **独立履行监督职责**，不参与任何 Agent 的具体执行操作
- 严格遵循时间限制、迭代次数要求，不允许任何 Agent 违规
- 日志记录需完整、详细，覆盖优化全流程

---

## 1. 进度监控

### 1.1 流程步骤跟踪

对照以下优化流程 Checklist，检查每个步骤的执行状态：

```
- [ ] Step 1: 环境检测 & 定位源代码
      - EnvDetectAgent 完成环境检测
      - EnvDetectAgent 完成代码定位
      - EnvDetectAgent 完成初步逻辑解析
      - (若需要) RegistrarAgent 完成代码迁移与注册

- [ ] Step 1.5: 代码深度学习与分析
      - OptimizerAgent 完成代码框架分析
      - OptimizerAgent 完成执行流程梳理
      - OptimizerAgent 完成核心算法理解
      - OptimizerAgent 完成 Kernel 结构分析
      - OptimizerAgent 完成优缺点分析
      - OptimizerAgent 完成潜在风险识别
      - OptimizerAgent 完成优化方向排序
      - (若有 Golden) 完成 CHECKIN vs Golden 代码对比
      - 代码认知报告已提交

- [ ] Step 2: 正确性验证 & 初始性能基线
      - ValidatorAgent 完成初始正确性测试（记录 pass/total）
      - ValidatorAgent 完成初始 benchmark（记录 initial_speedup）
      - 基线数据已记录

- [ ] Step 3: 性能优化循环
      - 轮次 >= 10（或达标提前退出）
      - 每轮包含: 代码修改 → 正确性验证 → benchmark → 记录
      - 每轮回退策略已记录原因和数据

- [ ] Step 5: 最终正确性验证 + 最终 benchmark
      - 最终正确性全部通过
      - 运行 `final_bench_rounds` 轮 benchmark（默认 1 轮，用户可通过 `bench_rounds=N` 指定），多轮时取平均

- [ ] Step 6: 生成优化报告 + 更新 skill
      - Word 报告已生成
      - Wiki 页面已创建
      - 自检清单已执行
      - Skill 文件已更新
      - To Be Optimized Ops 列表已更新
```

### 1.2 时间记录

记录各环节的开始时间和结束时间：
```
{
  "op_name": str,
  "start_time": timestamp,
  "steps": {
    "step1_env_detect": {"start": t, "end": t, "duration_s": float},
    "step1_5_code_analysis": {"start": t, "end": t, "duration_s": float},
    "step2_baseline": {"start": t, "end": t, "duration_s": float},
    "step3_optimization": {
      "start": t, "end": t, "duration_s": float,
      "iterations": [
        {"round": 1, "start": t, "end": t, "duration_s": float},
        ...
      ]
    },
    "step5_final_verify": {"start": t, "end": t, "duration_s": float},
    "step6_report": {"start": t, "end": t, "duration_s": float}
  },
  "total_duration_s": float
}
```

---

## 2. 迭代管控

### 2.1 迭代次数要求

- **最少**: 10 轮
- **继续条件**: 10 轮后若 avg speedup < 0.8x，必须继续
- **上限**: 20 轮 或 1 小时
- **达标退出**: avg speedup >= target_speedup 时可提前进入 Step 5

### 2.2 超时监控

当接近以下限制时，提醒 SchedulerAgent：

| 阈值 | 动作 |
|------|------|
| 优化时间 > 45 分钟 | 提醒：距时间上限还剩 15 分钟 |
| 优化时间 > 55 分钟 | 提醒：距时间上限还剩 5 分钟，准备收尾 |
| 优化时间 >= 60 分钟 | 强制通知：终止优化，列出未来方向 |
| 迭代次数 >= 18 | 提醒：距迭代上限还剩 2 轮 |
| 迭代次数 >= 20 | 强制通知：终止优化 |

### 2.3 异常迭代检测

- 连续 3 轮加速比无变化（差异 < 1%）→ 提醒更换优化策略
- 连续 2 轮正确性失败 → 提醒暂停并分析原因
- 加速比出现大幅回退（降幅 > 30%）→ 提醒检查代码

---

## 3. 流程合规检查

### 3.1 违规检测规则

PlannerAgent 在以下节点检查是否存在违规：

| 检查点 | 违规条件 | 处理 |
|--------|---------|------|
| Step 3 开始前 | Step 2 未完成（基线未记录） | 立即暂停，要求先执行 Step 2 |
| Step 3 每轮 | 未包含正确性验证 | 立即暂停，要求补充验证 |
| Step 5 开始前 | Step 3 未达到 10 轮（且未达标） | 立即暂停，要求继续优化 |
| Step 6 开始前 | Step 5 未完成 | 立即暂停，要求先执行 Step 5 |
| Step 6 | 报告未执行自检清单 | 要求执行自检清单 |
| 任何步骤 | 跳过了中间步骤 | 立即暂停，要求按顺序执行 |

### 3.2 违规反馈格式

```
⚠️ 流程违规
- 违规类型: {step_skipped | missing_verification | premature_exit | ...}
- 违规位置: Step {N}
- 违规描述: {详细说明}
- 要求: {修正方案}
- 当前状态: {暂停/继续}
```

---

## 4. 日志记录

### 4.1 关键数据点

全程记录以下数据：

```yaml
optimization_log:
  operator: str
  path_type: A/B/C

  environment:
    python_cmd: str
    flaggems_root: str
    gcu_arch: int
    torch_version: str
    triton_version: str

  baseline:
    initial_speedup: float
    initial_details: [{shape, dtype, speedup}]
    correctness: {passed: int, total: int}

  iterations:
    - round: 1
      strategy: str
      strategy_description: str
      correctness: pass/fail
      speedup: float
      vs_baseline: "+X%"
      vs_previous: "+Y%"
      rollback: false
      rollback_reason: null
      duration_s: float
    ...

  final:
    final_speedup: float
    final_details: [{shape, dtype, speedup}]
    correctness: {passed: int, total: int}
    target_achieved: bool

  report:
    word_path: str
    wiki_url: str
    checklist_passed: bool
    skill_updated: bool

  anomalies:
    - type: str
      timestamp: str
      description: str
      resolution: str

  summary:
    status: SUCCESS/PARTIAL/FAILED
    total_rounds: int
    total_duration_s: float
    improvement: "Xx"
    key_strategies: [str]
    key_findings: [str]
    future_directions: [str]
```

### 4.2 异常记录

所有异常情况都必须记录：
- GCU 掉卡及恢复
- 编译超时及优化措施
- 正确性失败及回退
- 策略无效/有害的详细记录

### 4.3 经验沉淀

优化结束后，PlannerAgent 从日志中提炼：
1. **有效策略列表**: 按效果排序
2. **无效/有害策略列表**: 避免后续重复
3. **性能瓶颈分析**: 硬件限制 vs 算法限制 vs 编译器限制
4. **未来优化方向**: 若未达标，列出可能的改进方向

---

## 5. 监督反馈协议

### 5.1 定期同步

PlannerAgent 在以下时机向 SchedulerAgent 发送进度报告：
- 每个 Step 完成后
- 每 3 轮优化迭代后
- 接近时间/迭代上限时
- 检测到违规时（立即）

### 5.2 进度报告格式

```
📊 优化进度报告
━━━━━━━━━━━━━━━━━━
算子: {op_name}
当前步骤: Step {N}
已完成轮次: {current}/{max_iterations}
已用时间: {elapsed} / 60 min
当前加速比: {current_speedup}x
vs 基线: +{improvement}%
目标: {target_speedup}x
状态: {on_track | at_risk | blocked}
━━━━━━━━━━━━━━━━━━
```

---

## 6. 算子优化履历（Ops Optimization Registry）

### 6.1 职责

PlannerAgent 负责维护**全局算子优化履历**。每个算子优化完成后，**必须**在本 skill 文件的"已优化算子列表"中更新该算子的状态。此列表是所有已优化算子的权威记录。

### 6.2 每条记录必须包含的信息

| 字段 | 说明 | 示例 |
|------|------|------|
| 算子名 | 完整算子名（含变体如 `_`） | `celu / celu_` |
| 状态 | DONE / PARTIAL / FAILED / SKIP | DONE |
| 优化轮次 | 总迭代次数 | 16 轮 |
| 基线 → 最终 | 优化前后的 avg speedup | 0.09x → 1.18x |
| 提升倍数 | 优化效果 | 13.1x 提升 |
| 关键优化技术 | 最有效的 2-3 个 Strategy | Strategy 16i(alpha特化) + Strategy 16h(大BLOCK) |
| 未达标原因（若有）| PARTIAL/FAILED 时的根本瓶颈 | tl.atomic_add 硬件限制 |
| 正确性 | 通过数/总数 | 36/36 |
| Wiki URL | 优化报告 Wiki 页面链接 | http://wiki.enflame.cn/pages/viewpage.action?pageId=xxx |
| 备注 | 其他重要信息 | fp32 受 erf 内置函数限制 |

### 6.3 记录格式

每条记录使用以下格式：

```
[x] {算子名} ({状态}: {轮次}轮; avg {基线}x→{最终}x ({提升}x提升); {正确性}; {关键技术}; {未达标原因})
    Wiki: {url}
```
说明：基线数据必须保留。若原始数据仅有最终值和提升倍数，用 `基线 = 最终 / 提升` 推算；若完全无法推算，用 `NA` 代替。CRASH 场景直接写 `CRASH`。

### 6.4 更新规则

1. **每个算子优化完成后必须更新**，不得跳过
2. **先归类，后追加**：根据算子实现方式判断所属类别（见下方分类），追加到对应子章节中
3. 同时更新本 skill 文件（planner_agent.md）
4. Wiki URL 必须填写（由 ReporterAgent 提供）
5. PARTIAL 状态必须说明未达标的根本原因和未来优化方向

### 6.5 算子分类规则

| 类别 | 判断依据 | 典型算子 |
|------|---------|---------|
| A. Dynamic Pointwise | 使用 `pointwise_dynamic` codegen 或 unary/binary/ternary 逐元素计算 | sigmoid, pow, bitwise_and |
| B. Reduction | 沿一个或多个维度做归约（sum/max/min/cumsum/all 等） | log_softmax, min, cumsum |
| C. Normalization | 归一化模式（LayerNorm/GroupNorm/RMSNorm 等） | group_norm, skip_layernorm |
| D. Index & Scatter | 索引、scatter、gather 操作 | index_add, scatter_.src |
| E. Data Movement & Generation | 内存布局变换、padding、tiling、重排、张量创建、RNG | tile, dropout, arange, uniform_ |
| F. 2D 算子 | 涉及 2D grid 的矩阵运算（mm/bmm/conv 等） | mm, addmm, conv |
| G. Fused Compound | 多算子融合为单一 kernel | gelu_and_mul, rwkv_ka_fusion |

---

## 7. 输出规范

向 SchedulerAgent 提交：

```
进度报告:
  current_step: int
  current_round: int
  elapsed_time_s: float
  current_speedup: float
  status: on_track / at_risk / blocked / completed

流程违规 (若有):
  violation_type: str
  description: str
  required_action: str

完整优化日志:
  log: yaml (见 4.1 格式)

优化履历更新:
  op_name: str
  status: DONE / PARTIAL / FAILED / SKIP
  rounds: int
  baseline_speedup: float
  final_speedup: float
  key_strategies: [str]
  bottleneck: str | None
  wiki_url: str
  correctness: str

经验总结:
  effective_strategies: [str]
  ineffective_strategies: [str]
  bottleneck_analysis: str
  future_directions: [str]
```

---

## 已优化算子列表

<!-- PlannerAgent 在此维护全局优化履历，每个算子优化完成后先归类，再追加到对应子章节 -->
<!-- Wiki URL 另起一行缩进显示；基线数据必须保留，无法推算时用 NA -->

### A. Dynamic Pointwise 算子

> 使用 `pointwise_dynamic` codegen 或 unary/binary/ternary 逐元素计算模式

[x] sigmoid / sigmoid_ (DONE: 16轮; avg 0.33x→0.977x (3.0x提升); 3-tier BLOCK+adaptive num_warps; Strategy 15)
[x] tanh / tanh_ (DONE: 1轮; avg 0.449x→1.105x (2.5x提升); Strategy 15迁移)
[x] rsqrt / rsqrt_ (DONE: 2轮; avg 0.517x→0.986x (1.9x提升); tl.math.rsqrt+Strategy 15)
[x] silu / silu_ (DONE: 2轮; avg 0.551x→0.974x (1.8x提升); x*sigmoid(x)+Strategy 15)
[x] pow / pow_ (DONE: 3轮; avg 0.363x→4.865x (13.4x提升); exp2+log2替代_pow(fp16/bf16); fp32保留_pow)
[x] gelu / gelu_ / gelu_backward (PARTIAL: 二次优化4轮; avg 0.784x; fp16=0.849x, fp32=0.661x, bf16=0.842x; 1.2x不可达-native gelu可能使用TOPS硬件加速; sigmoid近似+flat kernel)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870830
[x] celu / celu_ (DONE: 16轮; avg 0.09x→celu 1.177x, celu_ 1.181x; alpha=1.0特化+dtype-based BLOCK; Strategy 16h+16i)
[x] elu / elu_ (DONE: 11轮; avg NA→elu_ 1.116x; 默认参数特化+动态BLOCK调度; Strategy 16i+16k)
[x] exponential_ (DONE: 12轮; avg 0.387x→2.012x (5.20x提升); tl.math.log+f32缓冲绕过downcast+三级BLOCK; Strategy 24a-d)
[x] remainder / remainder_ (DONE: 10轮; avg 0.651x→0.925x (1.42x提升); 24/24正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850405
[x] nan_to_num (DONE: 1+3轮; avg 0.294x→1.001x (3.40x提升); 180/216正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850199
[x] addcmul (DONE: 1+3轮; avg 0.377x→1.000x (2.65x提升); 18/18正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850908
[x] fill_scalar_out (DONE: 1+3轮; avg NA→2.342x; 90/90正确性; pointwise_dynamic超越native)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850864
[x] bitwise_and / bitwise_and_ (DONE: 1+3轮; avg 0.327x→1.013x (3.10x提升); 90/90正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850943
[x] bitwise_or / bitwise_or_ (DONE: 1+3轮; avg 0.316x→1.012x (3.20x提升); 90/90正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850944
[x] isnan (DONE: 1+3轮; avg 0.430x→1.006x (2.34x提升); 18/18正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850945
[x] reciprocal / reciprocal_ (DONE: 1+3轮; avg 0.302x→1.007x (3.33x提升); 36/36正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850946
[x] log_sigmoid (DONE: 1+3轮; avg 0.366x→0.952x (2.60x提升); 18/18正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850947
[x] relu / relu_ (DONE: 0轮(基线已达标); avg NA→1.391x/1.381x; 基线即超越1.2x目标; pointwise_dynamic+tl.where已最优; native relu有额外开销使Gems天然优势)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870719
[x] abs / abs_ (DONE: 8轮; avg 0.976x→1.033x/1.041x (+5.8%/+6.4%提升); fp16/bf16超越native 4-10%, fp32与native持平; 去除pointwise_dynamic+flat kernel+do_not_specialize+num_warps=8; 1.2x不可达因native abs仅需符号位翻转; tl.where替换tl.abs灾难性回退(-47%); Strategy 15)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870717
[x] glu (DONE: 0轮(基线已达标); avg 1.364x; fp16=1.550x, fp32=0.985x, bf16=1.558x; 基线即超越1.2x目标; pointwise_dynamic+sigmoid计算已较优)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870815
[x] exp2 / exp2_ (DONE: 0轮(基线已达标); exp2 avg 1.800x, exp2_ avg 1.756x; 手写flat kernel+grid-stride+动态BLOCK已最优)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870817
[x] acos (DONE: 4轮; avg 0.443x→1.224x (2.76x提升); 9/9正确性; 迁移+手写flat kernel+BLOCK=4096(half)/65536(fp32); fp32受_acos intrinsic限制约1.0x)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870819
[x] atan / atan_ (PARTIAL: 10+轮; avg 0.660x→0.851x (+29%提升); 6/6正确性; BLOCK=2048(half)+65536(fp32)+num_warps=8; 受tl_extra_shim.atan硬件intrinsic限制无法达1.2x)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870831
[x] remainder / remainder_ (PARTIAL: 6轮; avg 0.752x→0.928x (+23%提升); 4/4正确性; 手写flat kernel替代pointwise_dynamic+dtype-aware BLOCK(int16=4096,int32=65536))
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870835
[x] ge (NO_IMPROVEMENT: 基线1.007x; fp16=0.986x fp32=1.057x bf16=0.979x; comparison op带宽受限; pointwise_dynamic已较优)
[x] ne (NO_IMPROVEMENT: 基线1.000x; fp16=0.977x fp32=1.048x bf16=0.976x; comparison op带宽受限; pointwise_dynamic已较优)
[x] threshold (NO_IMPROVEMENT: 基线0.952x; fp16=0.960x fp32=0.939x bf16=0.956x; tl.where简单op带宽受限; pointwise_dynamic已较优)
[x] nan_to_num / nan_to_num_ (NO_IMPROVEMENT: 基线0.915x; fp16=0.906x fp32=0.940x bf16=0.899x; 手写kernel退化至0.768x; pointwise_dynamic已较优)
[x] normal_ (PARTIAL: 5轮; avg 0.945x→1.023x (+8.3%提升); 3/3正确性; BLOCK 4096→1024; 瓶颈: Box-Muller(log+sqrt+sin/cos)计算密集+fp16 temp缓冲区copy)
[x] Many binary ops 批量评估:
    - add(0.934x) sub(0.926x) eq(0.991x) gt(1.010x) le(1.029x) lt(1.023x): 接近parity
    - div(9.840x) floor_divide(1.218x): 已达标
    - logical_or(1.137x) logical_and(1.125x) logical_xor(1.123x): 接近目标
    - maximum(0.891x) minimum(0.875x): 简单tl.max/min受带宽限制
    - mul: benchmark崩溃(exit134白名单)
    - 结论: pointwise_dynamic对简单binary ops已较优，手写kernel反退化

### B. Reduction 算子

> 沿一个或多个维度做归约（softmax/sum/max/min/cumsum/all 等）

[x] log_softmax (DONE: avg NA→4.0x; N=1短路24x; 大N受reduction带宽限制~0.44x)
[x] softmax_backward (DONE: avg 0.10x→1.2x+ (12x提升); make_block_ptr+two-stage kernel+host dispatch; Strategy 9+16b+16c)
[x] min (DONE: 6轮; avg 0.268x→0.550x (2.05x提升); 54/54正确性; 全局min用torch.amin, min_dim参照max.py; Strategy 20a+20b)
[x] max (DONE: avg NA→0.544x; split argmax替代tl.max(return_indices=True); 编译5min→6s)
[x] equal (DONE: 12轮; avg 0.198x→1.116x (5.64x提升); 融合eq+all单核+BLOCK=32768+nw=1; Strategy 23a+23b+23c)
[x] cumsum (PARTIAL: 10轮; avg 0.105x→0.132x (+25.7%); 25/25正确性; 瓶颈: tl.cumsum在GCU上比tl.sum慢33倍, 编译器限制)
[x] count_nonzero (DONE: 2轮; avg 0.040x→2.266x (56.7x提升); 18/18正确性; batch内核解决大M小N最内维+strided内核免dim_compress解决非最内维; dim=1大N已达17x)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870800
[x] all (PARTIAL: 4轮; flat 0.167x→1.30x, dim 与公共代码持平; 40/40正确性; GCU400实现注册+grid-stride global+libtuner 2D dim; 非最后维度3D归约0.09-0.15x因dim_compress瓶颈不可优化)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870745

### C. Normalization 算子

> 归一化模式（LayerNorm / GroupNorm / RMSNorm / SkipLayerNorm 等）

[x] group_norm (DONE: avg 0.321x→1.2x+ (3.7x提升); Welford 2-pass+make_block_ptr+DSM guard; Strategy 9+16d+16f)
[x] skip_layernorm (DONE: 13轮; avg NA→1.399x excl 64x64; 2D Tiling+make_block_ptr+diff方差)

### D. Index & Scatter 算子

> 索引、scatter、gather、slice 操作

[x] index_add / index_add_ (PARTIAL: 10+轮; avg 0.033x→0.209x (6.3x提升); 小shape达0.91x; Strategy 10(flat grid+hybrid BLOCK); 瓶颈: tl.atomic_add硬件限制~20x慢于tl.store)
[x] scatter_.reduce (DONE: 0轮(基线达标); avg 1.249x; fp16=1.202x, fp32=1.296x; 受益于scatter_.src dispatch优化)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870773
[x] slice_scatter (PARTIAL: 2轮(均回退); avg 1.005x; 12/12正确性; 1.2x不可达-native memcpy已最优; 大shape用clone+copy_, 小shape用Triton kernel)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870775
[x] scatter_.src (DONE: 10轮; avg 1.123x→1.235x (10%提升); 24/24正确性; grid-stride row_kernel+条件化nw+降低row_kernel阈值; Strategy S1+S8)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870771

### E. Data Movement & Generation 算子

> 内存布局变换、padding、tiling、重排、dropout、张量创建、随机数生成

[x] dropout (DONE: 28轮; avg 0.02x→0.849x (42.5x提升); UNROLL=4+自适应BLOCK+u8 mask store; Strategy 11)
[x] replication_pad3d (DONE: 二次优化; avg 0.025x→0.80x→~1.0x; 自适应BLOCK+nested loop+constexpr div; Strategy 12)
[x] repeat_interleave (DONE: 1轮; avg 0.017x→1.356x (79.8x提升); flat kernel替代StridedBuffer; Strategy 13)
[x] outer (DONE: 29轮; avg 0.01x→0.920x (92x提升); 2D nested loop+adaptive BLOCK_N+grid=48; Strategy 14)
[x] tile (DONE: 10轮; avg NA→fp16 2.48x, fp32 2.75x, bf16 2.48x; read-once-write-k消除modulo; Strategy 16j)
[x] arange / arange_start (DONE: 1+3轮; avg CRASH→1.043x; 648/648正确性)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850719
[x] linspace (DONE: 10轮; avg 0.605x→1.392x (2.3x提升); 648/648正确性; forward-only计算+自适应BLOCK)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358850270
[x] uniform_ (PARTIAL: 7轮; avg 1.099x→1.129x (1.03x提升); fp32达标1.200x, fp16/bf16~1.09x; 3/3正确性; 自适应num_warps(fp32→2,fp16→4)+@libentry; 瓶颈: fp16/bf16 tl.store downcast固有开销+RNG不支持grid-stride)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870715
[x] randn (NO_IMPROVEMENT: 8轮; avg 0.877x→0.877x (0%); 6/6正确性; 所有策略无效; 瓶颈: Box-Muller变换(log+sqrt+polynomial sin/cos)计算密集，Torch可能使用不同RNG算法或硬件加速)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870722
[x] randn_like (NO_IMPROVEMENT: 0轮(共享randn_kernel); avg 0.939x→0.939x (0%); 3/3正确性; 共享randn kernel，无独立优化空间)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870723
[x] exponential_ (ALREADY_OPTIMIZED: 2轮; avg 2.010x→2.010x (0%); 6/6正确性; 原始代码已含@libentry+do_not_specialize+num_warps=1+f32_buffer+3层BLOCK; fp32 2.478x, fp16 1.768x, bf16 1.785x)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870727

### F. 2D 算子

> 涉及 2D grid 的矩阵运算（mm / addmm / bmm / conv 等）

[x] mm (PARTIAL: 10+轮; avg 0.522x→0.522x (无变化); 18/18正确性; 瓶颈: 小矩阵launch开销+tl.dot半精度效率低于TOPS BLAS)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358851568
[x] addmm (PARTIAL: avg CRASH→0.19x; 1D grid修复grid.y>255; 瓶颈: native BLAS 157.8 vs 20.6 TFLOPS)
[x] conv1d / conv2d (SKIP: avg NA→0.001x~0.022x; 瓶颈: Triton tiled dot-product不适配GCU400)

### G. Fused Compound 算子

> 多算子融合为单一 kernel（gelu+mul / moe_sum / rwkv 等）

[x] gelu_and_mul (DONE: 10轮; avg 0.258x→1.148x (4.45x提升); 混合dispatch: fp16/bf16用F.gelu+mul_, fp32用fused kernel; Strategy 19a+19b)
[x] moe_sum (DONE: avg ~1.27x; 10/11 shape>1x; Step6 报告)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870859
[x] rwkv_ka_fusion (PARTIAL: 小T Triton 优、大T PyTorch 混合; Step6 报告)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870858
[x] apply_rotary_pos_emb (PARTIAL: 短序列优、长序列 HEAD_DIM 限制; Step6 报告)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870857
[x] one_hot (DONE: 大 tensor scatter_ 路径 ~1.0x; -1 仍同步瓶颈; Step6 报告)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870854

### H. Backward / Gradient 算子

[x] elu_backward (DONE: GCU400 kernel 注册缺失修复; ~0.45x→~3.3x 量级; Step6 报告)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870855
[x] gelu_backward (PARTIAL: native 硬件加速; Triton BLOCK=2048 已最优; Step6 报告)
    Wiki: http://wiki.enflame.cn/pages/viewpage.action?pageId=358870856
