# 📚 RAG 企业知识库问答系统

基于 **检索增强生成（RAG, Retrieval-Augmented Generation）** 的企业内部知识库智能问答系统。支持从多种格式文档构建向量索引，通过语义检索 + LLM 生成实现精准的上下文感知问答。

## 🏗 架构概览

```
离线入库：文档 → 加载 → 清洗 → 分块 → 嵌入 → 向量库
在线问答：用户问题 → 检索 → （重排）→ 构建提示词 → LLM 生成 → 答案
```

## ✨ 核心功能

- **多格式文档加载** — 支持 CSV、Markdown、PDF、TXT、Word (.docx) 五类格式，工厂模式自动适配 [loader](src/ingestion/loader.py:32)
- **文本清洗去噪** — 自动移除分隔线、页眉页脚、纯页码等噪声；压缩连续空白行 [cleaner](src/ingestion/cleaner.py:65)
- **语义分块** — 基于 `RecursiveCharacterTextSplitter` 递归切分，优先按段落 → 句子边界，支持可调 chunk_size / overlap [splitter](src/ingestion/splitter.py:18)
- **双模嵌入后端** — 支持 OpenAI Embedding API (`text-embedding-3-small`) 和本地 HuggingFace 模型 (如 `bge-small-zh`) [embedder](src/ingestion/embedder.py:117)
- **向量库可选** — Chroma（轻量持久化）或 FAISS（高性能检索），统一封装 [vectorizer](src/ingestion/vectorizer.py:82)
- **语义检索** — 向量相似度搜索，支持元数据过滤和分数阈值 [retriever](src/retrieval/retriever.py:40)
- **重排序增强** — 可选 BGE Cross-Encoder 对 Top-K 结果重排，提升相关块排名 [reranker](src/retrieval/reranker.py:68)
- **多场景提示词** — 精确查找 / 操作步骤 / 模糊推断，四套提示词模板自动检测匹配 [prompt_builder](src/generation/prompt_builder.py:167)
- **多 LLM 后端** — OpenAI / 百度文心 / 阿里通义千问 / 本地 vLLM，统一调用接口，内置指数退避重试 [llm_client](src/generation/llm_client.py:167)
- **Streamlit 可视化界面** — 对话式问答、检索来源展示、参数实时调节 [app.py](src/api/app.py:1)
- **环境变量配置** — YAML 中 `${VAR}` 占位符自动替换，敏感信息不入仓库 [config_loader](src/utils/config_loader.py:36)

## 🚀 快速开始

### 环境要求

- Python 3.10+
- pip

### 安装

```bash
# 克隆仓库
git clone <repo-url> && cd rag-enterprise-qa

# 创建虚拟环境（推荐）
python -m venv .venv && source .venv/bin/activate   # Linux/Mac
# 或: .venv\Scripts\activate                         # Windows

# 安装依赖
pip install -r requirements.txt
```

### 配置

编辑 `configs/app_config.yaml` 或设置环境变量：

```bash
# 使用 OpenAI（Embedding + LLM）
export OPENAI_API_KEY="sk-your-key"
# 可选：自定义 API 端点
export OPENAI_BASE_URL="https://your-proxy.com/v1"

# 使用百度文心
export QIANFAN_ACCESS_KEY="your-ak"
export QIANFAN_SECRET_KEY="your-sk"

# 使用阿里通义千问
export DASHSCOPE_API_KEY="your-key"
```

### 1. 准备数据

将文档放入 `data/raw/` 对应子目录：

```
data/raw/
├── csv/        # .csv 文件
├── markdown/   # .md 文件
├── pdf/        # .pdf 文件
├── txt/        # .txt 文件
└── word/       # .docx 文件
```

### 2. 构建索引

```bash
# 默认配置（Chroma + OpenAI）
python scripts/build_index.py

# 使用本地嵌入模型（无需 API Key）
python scripts/build_index.py --embedding-backend huggingface

# 使用 FAISS + 自定义分块参数
python scripts/build_index.py --backend faiss --chunk-size 800 --chunk-overlap 100

# 强制重建
python scripts/build_index.py --rebuild

# 只索引某一类文档
python scripts/build_index.py --category pdf
```

### 3. 问答测试

```bash
# 交互式问答
python scripts/test_qa.py

# 单次问答
python scripts/test_qa.py --query "CS210 产品的功率是多少？"

# 显示详细检索信息
python scripts/test_qa.py --verbose

# 批量处理
python scripts/test_qa.py --batch questions.txt --output answers.json
```

### 4. 启动 Web 界面

```bash
streamlit run src/api/app.py
```

浏览器访问 `http://localhost:8501`，即可使用可视化问答界面。

## 📖 详细用法

### 命令行参数

#### `scripts/build_index.py`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--raw-dir` | 原始文档目录 | `data/raw` |
| `--persist-dir` | 向量库持久化目录 | `data/vector_store` |
| `--backend`, `-b` | 向量库后端：`chroma` / `faiss` | 从配置读取 |
| `--chunk-size` | 分块大小（字符） | 从配置读取 |
| `--chunk-overlap` | 分块重叠（字符） | 从配置读取 |
| `--embedding-backend` | 嵌入后端：`openai` / `huggingface` | 从配置读取 |
| `--category`, `-c` | 只处理指定子目录 | 全部 |
| `--rebuild` | 强制重建索引 | 关闭 |
| `--no-validate` | 跳过向量验证 | 关闭 |
| `--verbose`, `-v` | 详细日志 | 关闭 |

#### `scripts/test_qa.py`

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--query`, `-q` | 单次问答文本 | - |
| `--batch`, `-b` | 批量处理输入文件 | - |
| `--output`, `-o` | 批量输出路径 | `{input}.json` |
| `--top-k`, `-k` | 检索结果数 | 从配置读取 |
| `--scenario`, `-s` | 场景模板 | 自动检测 |
| `--verbose`, `-v` | 显示检索详情 | 关闭 |

#### 交互模式命令

```
/scenario exact_lookup    # 切换场景模板
/topk 3                   # 调整检索数量
/quit                     # 退出
```

### 场景模板

系统根据问题特征自动选择模板，也可手动指定：

| 模板 | 触发词 | 适用场景 |
|------|--------|----------|
| `default` | 默认 | 通用问答 |
| `exact_lookup` | 多少、功率、参数、规格… | 精确数值查询 |
| `howto` | 怎么、如何、步骤、流程… | 操作步骤指导 |
| `fuzzy` | 类似、相关、大致… | 模糊匹配推断 |

## 📂 项目结构

```
rag-enterprise-qa/
├── configs/                        # 配置文件
│   ├── app_config.yaml             # 模型/参数配置
│   └── prompt_templates.yaml       # 提示词模板
├── data/
│   ├── raw/                        # 原始文档（按格式分目录）
│   ├── processed/                  # 清洗后文件
│   └── vector_store/               # 向量库持久化
├── src/
│   ├── ingestion/                  # 离线入库流水线
│   │   ├── loader.py               # 多格式加载器
│   │   ├── cleaner.py              # 文本清洗
│   │   ├── splitter.py             # 分块策略
│   │   ├── embedder.py             # 嵌入模型封装
│   │   └── vectorizer.py           # 向量库构建与持久化
│   ├── retrieval/                  # 检索模块
│   │   ├── retriever.py            # 向量检索
│   │   └── reranker.py             # 重排序
│   ├── generation/                 # 生成模块
│   │   ├── prompt_builder.py       # 提示词构建
│   │   └── llm_client.py           # LLM 调用客户端
│   ├── api/                        # Web 界面
│   │   └── app.py                  # Streamlit 应用
│   └── utils/                      # 工具模块
│       ├── config_loader.py        # 配置加载
│       └── logger.py               # 日志系统
├── scripts/                        # 可执行脚本
│   ├── build_index.py              # 离线索引构建
│   ├── test_qa.py                  # 交互式问答
│   ├── eval_retrieval.py           # 检索评估
│   └── eval_generation.py          # 生成评估
├── tests/                          # 单元测试
├── notebooks/                      # 实验笔记
├── requirements.txt
└── README.md
```

## ⚙ 配置说明

### 嵌入模型配置

```yaml
embedding:
  backend: openai              # openai | huggingface
  openai:
    model: text-embedding-3-small
    api_key: ${OPENAI_API_KEY}
    batch_size: 100
  huggingface:
    model_name: BAAI/bge-small-zh-v1.5
    device: null               # 自动选择 GPU/CPU
```

### LLM 配置

```yaml
llm:
  backend: openai              # openai | qianfan | dashscope | vllm
  openai:
    model: gpt-4o-mini         # 推荐性价比模型
    temperature: 0.0           # 事实性场景建议 0
    max_tokens: 2048
    timeout: 60
```

### 分块参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `chunk_size` | 500 | 分块的字符上限 |
| `chunk_overlap` | 50 | 相邻块重叠字符数 |
| `top_k` | 5 | 检索返回的候选块数 |
| `max_context_tokens` | 6000 | 提示词上下文上限 |

**调参建议**：若检索结果与问题不匹配 → 减小 `chunk_size`，增大 `top_k`；若答案丢失上下文 → 增大 `chunk_overlap`。

## 🧪 评估

```bash
# 检索命中率评估
python scripts/eval_retrieval.py

# 生成质量评估
python scripts/eval_generation.py
```

## 🗺 后续升级路线

- **增量入库** — 监控 `data/raw` 目录变化，自动更新向量库
- **混合检索** — BM25 关键词 + 向量相似度，可配置权重
- **用户反馈闭环** — 记录"赞/踩"，收集难例持续优化
- **权限控制** — 不同部门按文档范围隔离检索
- **多语言支持** — 英文嵌入模型及分块逻辑扩展

## 📄 License

MIT
