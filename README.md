# 🤖 Context-Aware Q&A System

基于 **LangChain** 和 **LangGraph** 构建的具有上下文能力的问答系统。

## ✨ 特性

- **上下文感知**：自动识别问题是否需要上下文支持
- **对话历史**：使用 LangGraph Checkpoint 存储对话历史
- **条件路由**：根据问题类型自动选择不同的处理流程
- **可扩展**：易于添加新的功能和集成

## 🛠️ 技术栈

- LangChain (LLM 调用)
- LangGraph (工作流编排)
- LangGraph Checkpoint Memory (对话记忆)
- OpenAI GPT 模型

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/cxl973/langchain-qa-system.git
cd langchain-qa-system
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 配置环境

```bash
cp .env.example .env
# 编辑 .env 填入你的 API Key
```

### 4. 运行

```bash
python app.py
```

## 📁 项目结构

```
langchain-qa-system/
├── app.py              # 主应用代码
├── requirements.txt    # 依赖
├── .env.example        # 环境变量示例
└── README.md           # 说明文档
```

## 🔧 工作流程

```
用户问题 → 分析问题 → [需要上下文?] 
                          ↓
                    是 → 检索上下文 → 生成答案
                    否 → 直接生成答案
```

## 🔌 支持的模型

### OpenAI (默认)
```bash
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-xxx
OPENAI_MODEL=gpt-4o-mini
```

### MiniMax
```bash
LLM_PROVIDER=minimax
MINIMAX_API_KEY=your-minimax-key
MINIMAX_MODEL=abab6.5s-chat
MINIMAX_API_BASE=https://api.minimax.chat/v1
```

> 获取 MiniMax API Key: https://platform.minimax.chat/

## 💻 编程使用

```python
from app import ContextAwareQA

qa = ContextAwareQA()

# 第一次对话
answer, thread = qa.ask_with_history("你好，请介绍一下自己")
print(answer)

# 继续对话（自动使用历史上下文）
answer2, _ = qa.ask_with_history("你刚才说的语言是什么?", thread)
print(answer2)
```

## 📝 License

MIT
