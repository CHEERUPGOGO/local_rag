# 智能法律助手 - RAG增强的法律问答系统

基于 **Ollama + Qwen2.5:7b** 的本地化法律问答系统，集成 RAG（检索增强生成）技术，提供专业、准确的法律咨询服务。

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ 功能特性

- 🎯 **RAG 知识增强** - 内置法律知识库，检索相关法条辅助回答
- 💬 **现代聊天界面** - 深色主题，类 DeepSeek 风格的用户体验
- 🔌 **RESTful API** - 支持跨设备、跨系统调用
- 📚 **可扩展知识库** - 支持动态添加法律文档
- 🚀 **本地部署** - 数据不出本地，隐私安全

## 📋 内置法律知识

| 法律类别 | 覆盖内容 |
|---------|---------|
| 劳动合同法 | 工资拖欠、解除合同、经济补偿 |
| 民法典-合同编 | 违约责任、损失赔偿 |
| 消费者权益保护法 | 退货退款、欺诈赔偿 |
| 婚姻家庭与继承 | 财产分割、遗产继承 |
| 侵权责任法 | 人身损害、交通事故赔偿 |

## 🚀 快速开始

### 1. 克隆项目

```bash
git clone https://github.com/CHEERUPGOGO/local_rag.git
cd local_rag
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

### 3. 安装 Ollama 并下载模型

**Windows/macOS:**
访问 [ollama.com](https://ollama.com) 下载安装

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

**下载模型:**
```bash
ollama pull qwen2.5:7b
```

### 4. 启动服务

```bash
python llama_fastapi.py
```

### 5. 访问应用

- 🌐 **聊天界面:** http://localhost:8000
- 📖 **API文档:** http://localhost:8000/docs

## 📡 API 接口

### 法律问答

```bash
POST /api/chat
Content-Type: application/json

{
    "prompt": "公司拖欠工资怎么办？",
    "use_rag": true,
    "stream": false
}
```

**响应示例:**
```json
{
    "question": "公司拖欠工资怎么办？",
    "answer": "根据《劳动合同法》第三十条...",
    "sources": ["劳动合同法"]
}
```

### 其他接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/api/chat` | POST | 法律问答（核心接口） |
| `/api/chat/stream` | GET | 流式输出问答 |
| `/api/knowledge/add` | POST | 添加法律知识 |
| `/api/knowledge/stats` | GET | 知识库统计 |
| `/api/knowledge/search` | GET | 搜索知识库 |

### 跨设备调用示例

```bash
# 使用 curl 调用
curl -X POST http://你的IP:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "网购假货如何索赔", "use_rag": true}'

# Python 调用
import requests
response = requests.post("http://你的IP:8000/api/chat", 
    json={"prompt": "离婚财产如何分割", "use_rag": True})
print(response.json())
```

## 📁 项目结构

```
local_rag/
├── llama_fastapi.py          # 主程序（含前端页面）
├── requirements.txt          # Python 依赖
├── legal_knowledge/          # 法律知识库
│   └── sample_laws.txt       # 示例法律文档
└── README.md
```

## 🔧 配置说明

### 切换模型

修改 `llama_fastapi.py` 中的模型名称：

```python
model="qwen2.5:7b"  # 可改为其他 Ollama 支持的模型
```

### 添加法律知识

**方式一：API 添加**
```bash
curl -X POST http://localhost:8000/api/knowledge/add \
  -H "Content-Type: application/json" \
  -d '{"content": "法律条款内容...", "source": "法律名称"}'
```

**方式二：修改代码**

在 `PRESET_LEGAL_KNOWLEDGE` 列表中添加新的法律文档。

## 🌐 局域网访问

服务启动后，同一网络下的其他设备可通过以下方式访问：

```
http://你的电脑IP:8000
```

查看本机IP：
- Windows: `ipconfig`
- Linux/Mac: `ifconfig` 或 `ip addr`

## ⚠️ 免责声明

本系统仅供学习和参考使用，不构成专业法律意见。如遇实际法律问题，请咨询专业律师。

## 📄 License

MIT License
