# Chunking 模块开发计划

## 目标

在 `tiny-rag` 中实现一个独立的 chunking 模块，输入解析后的 Markdown/text 与分块配置，输出带顺序、原文 rune offset、可选标题上下文的 chunk 列表。模块只负责切分，不负责 embedding、数据库、向量索引、召回、rerank 或最终 LLM 上下文拼接。

参考资料：

- `/Users/shenzhengyang/Documents/Obsidian Vault/RAG/WeKnora-Chunking-策略.md`
- `/Users/shenzhengyang/Documents/Obsidian Vault/RAG/WeKnora-Chunking-输入输出规约.md`
- `/Users/shenzhengyang/Documents/Obsidian Vault/RAG/WeKnora-三种切分器技术细节.md`

## 模块边界

### 输入

普通切分入口：

```go
Split(text string, cfg SplitterConfig) []Chunk
```

调试入口：

```go
SplitWithDiagnostics(text string, cfg SplitterConfig) ([]Chunk, *Diagnostics)
```

Parent-child 入口：

```go
SplitParentChild(text string, parentCfg SplitterConfig, childCfg SplitterConfig) ParentChildResult
```

`text` 是上游 docreader 已经解析好的整段 Markdown/text，可能来自 PDF、Word、Markdown、URL 等文件。

`SplitterConfig`：

```go
type SplitterConfig struct {
    ChunkSize    int
    ChunkOverlap int
    Separators   []string
    Strategy     string // auto / heading / heuristic / legacy / recursive
    TokenLimit   int
    Languages    []string
}
```

默认值：

- `ChunkSize = 512`
- `ChunkOverlap = 80`
- `Separators = []string{"\n\n", "\n", "。"}`
- `Strategy = ""` 时按 `legacy` 处理；只有显式 `"auto"` 才 profile 路由

### 输出

普通 chunk：

```go
type Chunk struct {
    Content       string
    ContextHeader string
    Seq           int
    Start         int
    End           int
}
```

关键约束：

- `Content` 必须是原文切片，不把标题面包屑写进去。
- `Start` / `End` 是 rune offset，不是 byte offset。
- `End - Start == len([]rune(Content))`。
- `Seq` 从 0 开始递增。
- `ContextHeader` 只用于 embedding 阶段临时拼接。

Embedding 文本由上层通过等价方法生成：

```text
ContextHeader

TrimSpace(Content)
```

如果上层有文档标题，可以再拼成：

```text
DocumentTitle
ContextHeader

TrimSpace(Content)
```

Parent-child 输出：

```go
type ParentChildResult struct {
    Parents  []Chunk
    Children []ChildChunk
}

type ChildChunk struct {
    Chunk
    ParentIndex int
}
```

Diagnostics：

```go
type Diagnostics struct {
    SelectedTier string
    TierChain    []string
    Rejected     []RejectedTier
    Profile      DocumentProfile
}
```

## 推荐代码结构

```text
internal/chunking/
  types.go              // Chunk、Config、Diagnostics、profile types
  strategy.go           // Split、SplitWithDiagnostics、tier chain、validation
  profile.go            // ProfileDocument、SelectStrategy
  splitter_legacy.go    // legacy recursive splitter
  splitter_heading.go   // Markdown heading splitter
  splitter_heuristic.go // PDF/plain-text heuristic splitter
  patterns.go           // 结构边界正则
  protected.go          // protected spans
  parent_child.go       // SplitParentChild
  token_budget.go       // TokenLimit / Languages 对 ChunkSize 的调整
```

如果项目语言不是 Go，也保持同等 API、字段语义和测试行为。

## 实现顺序

### 1. 基础类型与配置归一化

实现 `SplitterConfig`、`Chunk`、`ChildChunk`、`ParentChildResult`、`Diagnostics`。

配置归一化规则：

- `ChunkSize <= 0` 时补 `512`。
- public `Split` / `SplitWithDiagnostics` / `SplitParentChild` 入口中，`ChunkOverlap <= 0` 时补默认值 `80`。
- 直接调用底层 splitter，例如 `split_legacy` / `SplitText` 时，`ChunkOverlap = 0` 保留为零重叠。
- `ChunkOverlap` 上限为 `ChunkSize / 2`，超过时向下裁剪。
- `Separators` 为空时补 `["\n\n", "\n", "。"]`。
- `Strategy` 为空时按 `legacy`；显式 `"auto"` 才 profile 路由；未知 strategy 走 auto 分支。
- `TokenLimit` 存在时按语言估算字符预算，必要时下调 `ChunkSize`。

### 2. 实现 legacy splitter

legacy 是所有策略的兜底，也被 heading / heuristic 用来处理超大块。

实现要点：

- 先识别 protected spans，尽量不切开这些结构：
  - LaTeX 块：`$$ ... $$`
  - fenced code block：``` fenced ```
  - Markdown 图片：`![alt](url)`
  - Markdown 链接：`[text](url)`
  - Markdown 表格 header、separator、表格行
- 非保护区按 `Separators` 递归拆成 unit。
- 分隔符作为独立 piece 保留，重组时不丢原文结构。
- `mergeUnits` 按 `ChunkSize` 合并 unit，超过目标大小就 flush 当前 chunk。
- `computeOverlap` 从当前 chunk 尾部按 unit 取 overlap，满足：
  - `overlapLen + unitLen <= ChunkOverlap`
  - `overlapLen + unitLen + nextLen <= ChunkSize`
- overlap 开头去掉空白、纯分隔符、零宽 header marker。
- Markdown 表格按行作为不可拆 unit；大表跨 chunk 时尽量把 header + separator 作为合成前缀补到后续表格 chunk，但必须检查：
  - header 是否已经存在于 overlap 或下一个 unit。
  - header 与当前行列数是否匹配。
  - header + 当前 unit 是否能放进 `ChunkSize`。
- 表格 header tracker 使用 hook + priority 管理 active headers；多个 header 同时存在时按 priority 从高到低拼接。
- Markdown 表格 hook 默认优先级为 `15`，start pattern 匹配 `header row + separator row`，end pattern 匹配空白行或非表格段落。
- 空表头，例如 `||` + separator，会等待第一行数据，用第一行数据补全列名上下文。
- 表格段落断开、下一行开始新表、或列数变化时，会结束旧表头并在 merge 阶段先 flush 旧 chunk，避免两个表的 header 串用。
- 带合成表头的 chunk 中，`Content` 可能包含零宽度插入的 header；`Start/End` 仍指向该 chunk 实际覆盖的原文行范围。
- 单个 unit 超过绝对上限时强制硬切，绝对上限约 `7500` rune。
  - 优先在 `7500` 前最多 `200` 个字符内找换行或空格。
  - 找不到就按 `7500` 硬切。

验收重点：

- 不破坏 protected structures。
- 能产出 rune offset 正确的 chunk。
- `legacy` 永远可作为最后兜底。

### 3. 实现文档 profile 与 auto router

`auto` 不是 splitter，而是 tier chain 选择器。

`ProfileDocument(text)` 统计：

- Markdown 标题数量、标题密度、主导标题层级。
- 页分隔符 `\f`。
- 编号章节：`1. Intro`、`2.3 Methods`。
- 英文 / 德文 / 中文章节：`Chapter IV. Intro`、`Section 2. Details`、`Kapitel 1. Überblick`、`Abschnitt II: ...`、`第 3 节 方法`。
- 全大写短标题。
- 分隔线。
- 连续空行。
- 页脚：`Page 1 of 10`、`Seite 1 von 10`、`页码 2/9`、`页 8`。
- 表格、代码块。
- 粗略语言信号。

`SelectStrategy(profile)`：

- Markdown 标题明显时加入 `heading`：
  - 标题总数至少 `3`。
  - 标题密度大于 `0.005`。
  - 能找到主导标题层级。
- 启发式结构明显时加入 `heuristic`：
  - 结构标记总数至少 `5`；或
  - 存在页分隔符；或
  - 存在中英德章节标记。
- 永远追加 `legacy`。

显式 strategy：

- `heading`：`heading -> legacy`
- `heuristic`：`heuristic -> legacy`
- `legacy` / `recursive` / 空值：`legacy`
- `auto`：`ProfileDocument -> SelectStrategy`
- 未知值：按 `auto` 处理，即 `ProfileDocument -> SelectStrategy`

### 4. 实现 ValidateChunks 与 fallback

每个 tier 执行后都调用 `ValidateChunks`。通过就返回；失败就进入下一 tier，并把失败原因写入 diagnostics。

校验规则：

- 不能没有 chunk。
- 大文档不能只产出一个 chunk：
  - `totalChars > 2 * chunkSize && len(chunks) == 1`
- 不能切得太碎：
  - 除最后一个 chunk 外，小于 `50` 字的 tiny chunk 过多。
  - `tinyCount > len(chunks)/4 && tinyCount > 2`
- 不能整体远低于目标大小：
  - `maxLen < chunkSize/4 && totalChars > chunkSize`
- 不能严重超过目标大小：
  - `maxLen > 2 * chunkSize`

### 5. 实现 heading splitter

适合结构清晰的 Markdown。

实现要点：

- 找主导标题层级：
  - 优先选择出现次数至少 `3` 次的最浅标题层级。
  - 否则选择文档里存在的最深标题层级。
  - 完全没有标题或 heading split 只能形成单个 section 时，直接落到 legacy。
- 识别 Markdown ATX 标题：
  - 正则形态：`^(#{1,6})\s+...`
  - 必须整行匹配。
  - 扫描时维护 `inFence`，代码块里的 `#` 不算标题。
- boundary 包括文档开头与所有 `level <= primaryLevel` 的标题位置。
- 每两个 boundary 之间形成一个 section。
- 使用 `HeadingHierarchy` 按 section 顺序推进标题状态，而不是每次按 offset 回查完整标题列表。
  - 进入 section 时先 observe 当前 boundary heading。
  - 立即 snapshot 当前 breadcrumb，作为短 section 的 `ContextHeader`。
  - 然后 observe section 内 `level > primaryLevel` 的 sub-heading，让 hierarchy 为后续 section 保持同步。
- section 能放进 `ChunkSize` 时直接产出 chunk：
  - `Content = section 原文`
  - `ContextHeader = section boundary 处 snapshot 的标题面包屑`
  - `Start/End = 原文 rune offset`
- section 过长时调用 legacy 递归切，并把 sub chunk 的 offset 加上 section 起点。
- 长 section 内部额外构建 `BreadcrumbPoint` 列表：
  - 起点使用 section boundary breadcrumb。
  - 遇到更深 sub-heading 时推进 hierarchy 并记录该 offset 的 breadcrumb。
  - 每个 sub chunk 使用其 start offset 处生效的最深 breadcrumb。
- 最后合并相邻小块 `coalesceTinyChunks`，条件：
  - 两个 chunk 有共同标题前缀。
  - `cur.End == next.Start`。
  - `curLen < target`，其中 `target = max(ChunkSize/2, 200)`。
  - `curLen + nextLen <= ChunkSize`。

### 6. 实现 heuristic splitter

适合 PDF 转文本或纯文本。

候选 boundary：

- `\f` 页分隔符。
- 编号标题：`1. Introduction`、`2.3 Methods`。
- 罗马数字标题：`IV. Results`。
- 英文章节：`Chapter 1. Overview`、`Section IV: Details`。
- 德文章节：`Kapitel 1. Überblick`、`Abschnitt II: Details`、`Teil 3 ...`。
- 中文章节：`第一章 项目背景`、`第 3 节 方法`、`第 1 部分 ...`。
- 全大写短标题。
- 分隔线。
- 页脚：`Page 1 of 10`、`Seite 1 von 10`、`页码 2/9`、`页 8`。
- 连续三个以上换行：`\n{3,}`。

实现要点：

- 逐行扫描时维护 `inFence`，不在 fenced code block 内识别结构边界。
- 如果 `Languages` 明确指定语言，则章节 marker 只启用对应语言的英文、德文或中文 pattern；未指定或无法识别时启用全部。
- 复用 legacy 的 protected spans。
- 将 protected spans 转为 rune offset 后，删除严格落在保护区内部的 boundary；边缘 boundary 保留。
- boundary 排序去重，同位置保留优先级高者。
- 推荐优先级：
  - form feed
  - numbered heading
  - chapter marker
  - all-caps heading
  - visual separator
  - page footer
  - blank block
- 相邻 boundary 形成 block 后，按 `ChunkSize` 贪心装箱。
- `minChunkSize = max(ChunkSize/4, 50)`。
- 如果加入下一个 block 会超，且当前累计长度达到 `minChunkSize`，flush 当前 chunk。
- 单个 block 超过 `ChunkSize` 时调用 legacy 切分。
- flush 后下一个 chunk 的起点使用 `applyOverlapAligned`：
  - `target = curEnd - ChunkOverlap`
  - `windowStart = curEnd - 2 * ChunkOverlap`
  - 优先在 `[windowStart, curEnd)` 找最近语义 boundary。
  - 其次从 `target` 往前找最近换行，返回换行后一位。
  - 最后才用原始 `target`。

### 7. 实现 parent-child

Parent-child 是执行模式，不是 strategy。

流程：

1. 使用 `parentCfg` 对全文切 parent chunks。
2. 对每个 parent 的 `Content` 使用 `childCfg` 切 child chunks。
3. child 的 `Start/End` 要加上 parent 的 `Start`，映射回全文 rune offset。
4. 只有 parent 被 child 实际拆开或改写时，才把 parent 保留到 `Parents`，并让 child 的 `ParentIndex` 指向保留后的 parent index。
5. 如果 parent 未被实际拆开，即 child 只有一个且 `child.Content == parent.Content`，则不保留 parent，child 的 `ParentIndex = -1`。
6. child 的 `ContextHeader` 由 parent/child 面包屑合并得到；如果 parent 最后一行和 child 第一行 trim 后相同，要去掉 child 的重复 heading。

默认建议：

- parent chunk size 约 `4096`。
- child chunk size 约 `384`。
- child 负责向量匹配，被实际保留的 parent 负责回答上下文。

注意：

- parent-child 应复制 `Strategy`、`TokenLimit`、`Languages`，避免父子切分意外退化成 legacy。
- parent-child 作为 public 入口时，`ChunkOverlap <= 0` 会按默认值 `80` 归一化。

### 8. 实现 preview / debug 输出

`SplitWithDiagnostics` 需要返回：

- 最终采用的 tier。
- 本次尝试的 tier chain。
- 被拒绝 tier 与原因。
- auto router 的 profile。

Preview 层可以额外计算：

- 每个 chunk 的字符数。
- 每个 chunk 的近似 token 数。
- chunk size 的 avg / min / max / stddev。
- `ContextHeader`。
- 内容预览。

## 测试矩阵

### 单元测试

- `SplitterConfig` 默认值、`ChunkOverlap <= 0`、`TokenLimit` 下调。
- rune offset：中英文混排、emoji、中文标点、换行。
- protected spans：代码块、LaTeX、图片、链接、Markdown 表格不被切坏。
- legacy：
  - 段落 / 换行 / 中文句号递归拆分。
  - overlap 不超过限制且不以纯分隔符开头。
  - 表格跨 chunk 自动补 header。
  - 超长无空格文本按约 `7500` rune 强切。
- heading：
  - 代码块内 `#` 不当作标题。
  - 主导标题层级选择正确。
  - section 过长时 fallback legacy。
  - 深层标题面包屑正确。
  - 相邻 tiny chunks 只在安全条件下合并。
- heuristic：
  - PDF 页分隔符、编号标题、Chapter/Section/Kapitel/中文章节识别。
  - protected span 内 boundary 被丢弃。
  - boundary 去重优先级正确。
  - 超大 block fallback legacy。
  - overlap 起点对齐语义 boundary 或换行。
- auto router：
  - Markdown 文档选择 `heading -> legacy`。
  - PDF 转文本选择 `heuristic -> legacy`。
  - 混合文档选择 `heading -> heuristic -> legacy`。
  - 普通长文本选择 `legacy`。
- validation：
  - 空结果失败。
  - 大文档单 chunk 失败。
  - tiny chunks 过多失败。
  - 全部 chunk 远小于目标失败。
  - 超过 `2 * ChunkSize` 失败。
- parent-child：
  - child offset 映射回全文。
  - `ParentIndex` 正确。
  - parent 不进入 embedding 的约定由上层使用，但结构要能表达。

### 集成测试样例

- Markdown 产品手册：多层 `# / ## / ###` 标题、代码块、表格。
- PDF 转文本：`\f`、页脚、编号章节、连续空行。
- 中文报告：`第一章`、`一、`、中文句号、中文表格说明。
- FAQ / 结构化短问答：验证可配置零 overlap。
- 超长连续文本：验证绝对最大长度保护。

## 验收标准

- `Split` 对任意非空文本最终能产出稳定 chunks。
- 所有 chunk 的 `Start/End/Content` 与原文 rune 切片一致。
- `Content` 不包含额外注入的 `ContextHeader`。
- `auto` 能按文档结构生成合理 tier chain，并由 `legacy` 兜底。
- `heading` 能保留章节面包屑，并在长 section 内继续细分。
- `heuristic` 能利用 PDF/纯文本结构边界，并避免在 protected span 内切分。
- `legacy` 能保护常见 Markdown 特殊结构，并处理超长异常输入。
- `SplitParentChild` 能返回 parent chunks 与带 `ParentIndex` 的 child chunks。
- `SplitWithDiagnostics` 能解释最终选择、fallback 原因和 profile 结果。
- 核心测试覆盖 legacy、heading、heuristic、auto router、validation、parent-child、rune offset。
