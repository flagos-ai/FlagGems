# 故障分类与修复边界

## 名称与实际调用

先从当前源码填映射表，不从文件名或 Python 运算符拼接函数名：

| 身份/入口 | 查验位置 |
|---|---|
| 任务 operator、Definition.name、输入/输出及 effects | 当前任务的原始 Definition；确认来源 |
| pytest marker、实际文件、别名 | 正确性/benchmark 装饰器，当前 KGS discovery |
| benchmark op_name、native overload | benchmark 构造参数与实际 list-cases 输出 |
| override key、实际被调用的 callable | cli_override、dynamic registry、准确性测试调用路径及逐次调用记录 |
| 内置实现/导出/注册 | ops 文件、ops/__init__.py、flag_gems/__init__.py 与后端覆盖 |
| 输入 ABI | 参数顺序/关键字、布局、dtype、标量、mutation/alias、输入 recipe |

marker 仅选择 pytest 节点，不能改变 benchmark 返回的 operator，也不能自动接通 candidate。KGS 可能区分公共身份 `native_operator` 与 `pytest_marker`；读取当前实现，再判断何处不一致。当前修复契约要求 Definition、benchmark operator、override key 一致，允许 marker 和实际文件名使用已验证别名。

- 单前导下划线按本项目约定，如 `_add` 使用 `underscore_add`；同步自身正确性、benchmark 和实际 discovery。
- dunder **不能照此猜下划线数量**。本次已验证 `__iand__` / `__irshift__` / `__ixor__` 使用 `iand` / `irshift` / `ixor` marker，operator/override 保留原 dunder 名。不要据此假定所有分支采用同一转换算法。
- `flag_gems.irshift`、`flag_gems.__irshift__`、`flag_gems.bitwise_right_shift_` 是三个不同查找路径。语义相同不等于同一个注入入口；“没有 irshift 导出”不证明没有 `__irshift__` 实现。
- 找不到 benchmark 可能是 marker、真实路径、过期 checkout、Definition 身份或 discovery 失败，先分清，不默认复制一个文件。
- 给共用函数叠加 marker 前，确认选中后所有 benchmark 的 op_name 均属于该 Definition；不能把其他 overload 的 catalog 混进来。共享文件可以保留，不必按 overload 拆文件。

### 正确性注入

优先直接调用对应、已存在的 `flag_gems` 导出。当前 registry 若允许为缺失属性建立 override，且没有统一导出，可在自身文件里做最小按调用时查找的 dispatcher：先取已注入的原身份，仅在属性缺失时选自身已有实现。Tensor/Scalar 分发是实现选择，不要求任务必须改用两种实现名。

不要在模块导入时缓存 candidate；不要在候选抛异常后静默调用 reference；不要用 `use_gems()` 替换路径、子模块预绑定或不同导出绕过原 override key。仅当当前实际路径确实绕过注入时修它，不全库机械替换调用。

候选名字/函数名或 Definition 错误，与 Gems 测试、注册缺陷分列；按用户分工决定反馈给 candidate 作者还是修正本算子的 Definition；自身 Definition 在三件套白名单中，其他算子的契约不顺带改，不随意改候选或弱化 Definition 来掩盖测试问题。

## 输入构造/签名问题

- 追踪 CaseSpec 外层与 BenchmarkCasePlan 内层各一层。`builder_args=(shape,)` 是一项 tuple；先取 `(shape,) = plan.builder_args`，再拆 recipe 的各字段。构造真实输入验证尺寸、stride、offset、存储边界及语义，不能只证明 unpack 不报错。
- 区分 out 为独立 overload。公开 `torch.foo` 可能没有 `out=`，而 `torch.ops.aten.foo.out` 有；检查当前运行时真实 schema 和调用，保留关键字参数、generator、别名和原地契约。
- benchmark 构造期间访问缺失 `flag_gems.XXX` 会先于 override 失败。删显式 eager gems_op，不改公共 runner。
- 部分历史 catalog 的 shape 是 recipe 描述，不是实际 Tensor 维度。保留原 case 契约并核对真实 builder；若确需改 catalog 语义，显式说明并重验对应 contract，不能悄悄改值后宣称原组合未变。

## 后端 reference 的部分能力缺失

用合法小输入复现具体失败，再跑完整原参数集；记录 backend、实际 dtype/shape/标量、API 和模式。排除输入协议、环境和测试 bug，才加 precise skip。

- 默认 reference 失败不授权 CPU-reference skip。反之 CPU 也不支持某 dtype，要单独证明；不要为全绿删除 dtype/shape 或只保留 quick。
- 用条件 skip 保留原参数/case 列举；每个模式分别列出支持项、跳过项及理由。case-builder/body skip 要在 catalog 中仍能列出原 core case，不能伪造未执行计时。
- SDK 可能静默把请求 Double 变为 Float32。显式检查实际输入和输出 dtype；CPU 构造/字节搬运只能在验证无损后局部使用，不能偷换请求精度。
- reference 有厂商 padding 时，先用独立原生/CPU 结果确认逻辑契约，再仅规范 reference。禁止切片、cast 或搬运 candidate 以掩盖错误 device/dtype/shape；这些先断言再做数值观察。
- 不宽化容差、移除异常测试或用 broad except/xfail 隐藏故障。合法期望异常用例保留；dummy 被合法 RuntimeError 捕获时也要靠直接调用记录识别命中。

## 整个 reference API 缺失

先用合法小输入排除输入协议、环境与签名问题，再确认当前后端同 API/overload 的原生能力。API 部分返回不代表可用，例如 forward 返回 hy 但 workspace=None，仍不能 bootstrap backward。

若整个 API 不可用，记录实际 schema、最小复现、错误、后端/运行时和受影响 case，标记“暂不处理”，继续其他算子。不新增 CPU oracle、不采集或接入其他机器基线、不删除整组 case 或全部 skip 后宣称已修复。CPU 模式仍作为受支持算子的既有数值比较模式，不代表扩大本 skill 到整体缺失 reference 的替代方案。
