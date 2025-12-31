# 导入所需核心模块
from fastapi import FastAPI, Form, HTTPException, Body
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Generator
import uvicorn
import ollama
import os
import json

# ================= RAG 相关模块 =================
# 简易向量存储（生产环境建议使用ChromaDB/Faiss/Milvus）
import hashlib
from difflib import SequenceMatcher

class LegalKnowledgeBase:
    """法律知识库 - 简易RAG实现"""
    
    def __init__(self):
        self.documents = []
        self.chunks = []
        
    def add_document(self, content: str, source: str = "unknown"):
        """添加法律文档到知识库"""
        doc_id = hashlib.md5(content.encode()).hexdigest()[:8]
        # 分块处理（每块约500字）
        chunk_size = 500
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        
        for i, chunk in enumerate(chunks):
            self.chunks.append({
                "id": f"{doc_id}_{i}",
                "content": chunk,
                "source": source
            })
        self.documents.append({"id": doc_id, "source": source, "content": content})
        return doc_id
    
    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """基于关键词相似度检索相关法律知识"""
        if not self.chunks:
            return []
        
        results = []
        for chunk in self.chunks:
            # 简易相似度计算（生产环境建议使用向量相似度）
            similarity = SequenceMatcher(None, query, chunk["content"]).ratio()
            # 关键词匹配加分
            keywords = query.replace("？", "").replace("?", "").split()
            keyword_score = sum(1 for kw in keywords if kw in chunk["content"]) / max(len(keywords), 1)
            final_score = similarity * 0.4 + keyword_score * 0.6
            results.append({**chunk, "score": final_score})
        
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]
    
    def get_stats(self) -> dict:
        """获取知识库统计信息"""
        return {
            "documents_count": len(self.documents),
            "chunks_count": len(self.chunks)
        }

# 初始化知识库
knowledge_base = LegalKnowledgeBase()

# 预置法律知识
PRESET_LEGAL_KNOWLEDGE = [
    {
        "source": "劳动合同法",
        "content": """《中华人民共和国劳动合同法》关键条款：
第三十条：用人单位应当按照劳动合同约定和国家规定，向劳动者及时足额支付劳动报酬。用人单位拖欠或者未足额支付劳动报酬的，劳动者可以依法向当地人民法院申请支付令，人民法院应当依法发出支付令。
第三十八条：用人单位有下列情形之一的，劳动者可以解除劳动合同：（一）未按照劳动合同约定提供劳动保护或者劳动条件的；（二）未及时足额支付劳动报酬的；（三）未依法为劳动者缴纳社会保险费的。
第八十五条：用人单位有下列情形之一的，由劳动行政部门责令限期支付劳动报酬、加班费或者经济补偿；劳动报酬低于当地最低工资标准的，应当支付其差额部分；逾期不支付的，责令用人单位按应付金额百分之五十以上百分之一百以下的标准向劳动者加付赔偿金。"""
    },
    {
        "source": "民法典-合同编",
        "content": """《中华人民共和国民法典》合同编要点：
第五百零九条：当事人应当按照约定全面履行自己的义务。当事人应当遵循诚信原则，根据合同的性质、目的和交易习惯履行通知、协助、保密等义务。
第五百七十七条：当事人一方不履行合同义务或者履行合同义务不符合约定的，应当承担继续履行、采取补救措施或者赔偿损失等违约责任。
第五百八十四条：当事人一方不履行合同义务或者履行合同义务不符合约定，造成对方损失的，损失赔偿额应当相当于因违约所造成的损失。"""
    },
    {
        "source": "消费者权益保护法",
        "content": """《中华人民共和国消费者权益保护法》核心内容：
第七条：消费者在购买、使用商品和接受服务时享有人身、财产安全不受损害的权利。
第二十四条：经营者提供的商品或者服务不符合质量要求的，消费者可以依照国家规定、当事人约定退货，或者要求经营者履行更换、修理等义务。
第五十五条：经营者提供商品或者服务有欺诈行为的，应当按照消费者的要求增加赔偿其受到的损失，增加赔偿的金额为消费者购买商品的价款或者接受服务的费用的三倍；增加赔偿的金额不足五百元的，为五百元。"""
    },
    {
        "source": "婚姻法与继承法",
        "content": """《民法典》婚姻家庭编与继承编要点：
第一千零六十二条：夫妻在婚姻关系存续期间所得的下列财产，为夫妻的共同财产，归夫妻共同所有：（一）工资、奖金、劳务报酬；（二）生产、经营、投资的收益；（三）知识产权的收益；（四）继承或者受赠的财产。
第一千零七十九条：夫妻一方要求离婚的，可以由有关组织进行调解或者直接向人民法院提起离婚诉讼。人民法院审理离婚案件，应当进行调解；如果感情确已破裂，调解无效的，应当准予离婚。
第一千一百二十七条：遗产按照下列顺序继承：（一）第一顺序：配偶、子女、父母；（二）第二顺序：兄弟姐妹、祖父母、外祖父母。"""
    },
    {
        "source": "侵权责任法",
        "content": """《民法典》侵权责任编要点：
第一千一百六十五条：行为人因过错侵害他人民事权益造成损害的，应当承担侵权责任。
第一千一百七十九条：侵害他人造成人身损害的，应当赔偿医疗费、护理费、交通费、营养费、住院伙食补助费等为治疗和康复支出的合理费用，以及因误工减少的收入。造成残疾的，还应当赔偿辅助器具费和残疾赔偿金；造成死亡的，还应当赔偿丧葬费和死亡赔偿金。
第一千二百一十三条：机动车发生交通事故造成损害，属于该机动车一方责任的，先由承保机动车强制保险的保险人在强制保险责任限额范围内予以赔偿。"""
    }
]

# 初始化预置知识
def init_knowledge_base():
    for doc in PRESET_LEGAL_KNOWLEDGE:
        knowledge_base.add_document(doc["content"], doc["source"])
    print(f"✅ 法律知识库初始化完成: {knowledge_base.get_stats()}")

init_knowledge_base()

# ================= FastAPI 应用初始化 =================
app = FastAPI(
    title="智能法律助手 API",
    description="基于RAG增强的Qwen2.5:7b法律问答系统，支持跨设备、跨系统调用",
    version="2.0.0"
)

# CORS配置 - 支持跨域调用
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= 数据模型 =================
class ChatRequest(BaseModel):
    """聊天请求模型"""
    prompt: str
    use_rag: bool = True
    stream: bool = False
    history: Optional[List[dict]] = None

class ChatResponse(BaseModel):
    """聊天响应模型"""
    question: str
    answer: str
    sources: Optional[List[str]] = None
    rag_context: Optional[str] = None

class KnowledgeAddRequest(BaseModel):
    """添加知识请求模型"""
    content: str
    source: str = "用户上传"

# ================= 核心功能 =================
def call_llama31_with_rag(prompt: str, use_rag: bool = True, history: list = None) -> tuple:
    """
    调用Llama3.1:8b模型，支持RAG增强
    返回：(回答内容, 相关法律来源列表, RAG上下文)
    """
    sources = []
    rag_context = ""
    
    if use_rag:
        # 检索相关法律知识
        relevant_docs = knowledge_base.search(prompt, top_k=3)
        if relevant_docs:
            rag_context = "\n\n".join([f"【{doc['source']}】\n{doc['content']}" for doc in relevant_docs])
            sources = list(set([doc['source'] for doc in relevant_docs]))
    
    # 构建增强提示词
    system_prompt = """你是一位专业的中国法律顾问AI助手。请基于提供的法律知识和你的专业知识，为用户提供准确、专业、易懂的法律解答。
回答要求：
1. 引用具体法律条款时请注明出处
2. 给出实际可操作的建议
3. 如遇复杂情况，建议用户咨询专业律师
4. 语言简洁明了，避免过于晦涩的法律术语"""

    messages = [{"role": "system", "content": system_prompt}]
    
    # 添加历史对话
    if history:
        messages.extend(history[-6:])  # 保留最近3轮对话
    
    # 构建用户消息
    if rag_context:
        user_content = f"""参考以下法律知识：
{rag_context}

用户问题：{prompt}

请基于以上法律知识和你的专业判断，给出详细解答。"""
    else:
        user_content = f"用户法律问题：{prompt}\n\n请提供专业的法律解答。"
    
    messages.append({"role": "user", "content": user_content})
    
    try:
        response = ollama.chat(model="qwen2.5:7b", messages=messages)
        return response["message"]["content"], sources, rag_context
    except Exception as e:
        error_msg = f"调用模型失败：{str(e)}。请确认Ollama服务已启动且llama3.1:8b模型已安装。"
        return error_msg, [], ""

def stream_llama31_with_rag(prompt: str, use_rag: bool = True, history: list = None) -> Generator:
    """流式输出版本"""
    sources = []
    rag_context = ""
    
    if use_rag:
        relevant_docs = knowledge_base.search(prompt, top_k=3)
        if relevant_docs:
            rag_context = "\n\n".join([f"【{doc['source']}】\n{doc['content']}" for doc in relevant_docs])
            sources = list(set([doc['source'] for doc in relevant_docs]))
    
    system_prompt = """你是一位专业的中国法律顾问AI助手。请基于提供的法律知识和你的专业知识，为用户提供准确、专业、易懂的法律解答。"""
    
    messages = [{"role": "system", "content": system_prompt}]
    if history:
        messages.extend(history[-6:])
    
    if rag_context:
        user_content = f"参考法律知识：\n{rag_context}\n\n用户问题：{prompt}"
    else:
        user_content = f"用户法律问题：{prompt}"
    
    messages.append({"role": "user", "content": user_content})
    
    try:
        # 先发送来源信息
        if sources:
            yield f"data: {json.dumps({'type': 'sources', 'data': sources}, ensure_ascii=False)}\n\n"
        
        # 流式输出回答
        stream = ollama.chat(model="qwen2.5:7b", messages=messages, stream=True)
        for chunk in stream:
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield f"data: {json.dumps({'type': 'content', 'data': content}, ensure_ascii=False)}\n\n"
        
        yield f"data: {json.dumps({'type': 'done'}, ensure_ascii=False)}\n\n"
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'data': str(e)}, ensure_ascii=False)}\n\n"

# ================= API 接口 =================

@app.post("/api/chat", response_model=ChatResponse, summary="法律问答接口（JSON）")
async def api_chat(request: ChatRequest):
    """
    RESTful API接口 - 支持跨设备调用
    
    - **prompt**: 法律问题
    - **use_rag**: 是否使用知识库增强（默认True）
    - **stream**: 是否流式输出（默认False）
    - **history**: 历史对话记录
    """
    if request.stream:
        return StreamingResponse(
            stream_llama31_with_rag(request.prompt, request.use_rag, request.history),
            media_type="text/event-stream"
        )
    
    answer, sources, rag_context = call_llama31_with_rag(
        request.prompt, request.use_rag, request.history
    )
    
    return ChatResponse(
        question=request.prompt,
        answer=answer,
        sources=sources if sources else None,
        rag_context=rag_context if rag_context else None
    )

@app.get("/api/chat/stream", summary="流式问答接口")
async def api_chat_stream(prompt: str, use_rag: bool = True):
    """流式输出接口 - 适用于实时显示"""
    return StreamingResponse(
        stream_llama31_with_rag(prompt, use_rag),
        media_type="text/event-stream"
    )

@app.post("/api/knowledge/add", summary="添加法律知识")
async def add_knowledge(request: KnowledgeAddRequest):
    """向知识库添加新的法律文档"""
    doc_id = knowledge_base.add_document(request.content, request.source)
    return {"success": True, "doc_id": doc_id, "stats": knowledge_base.get_stats()}

@app.get("/api/knowledge/stats", summary="知识库统计")
async def get_knowledge_stats():
    """获取知识库统计信息"""
    return knowledge_base.get_stats()

@app.get("/api/knowledge/search", summary="搜索知识库")
async def search_knowledge(query: str, top_k: int = 5):
    """搜索知识库中的相关法律条款"""
    results = knowledge_base.search(query, top_k)
    return {"query": query, "results": results}

# 表单接口（兼容旧版）
@app.post("/llama/legal/chat", summary="法律问答接口（表单）")
async def llama_legal_chat(prompt: str = Form(...)):
    """接收HTML表单提交的法律问题"""
    answer, sources, _ = call_llama31_with_rag(prompt)
    return {"你的提问": prompt, "模型回答": answer, "参考来源": sources}

# ================= 前端页面 =================

@app.get("/", response_class=HTMLResponse, summary="智能法律助手页面")
async def chat_page():
    """DeepSeek风格的现代聊天界面"""
    return """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>智能法律助手 - RAG增强</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        :root {
            --bg-primary: #1a1a2e;
            --bg-secondary: #16213e;
            --bg-tertiary: #0f0f1a;
            --accent: #4f46e5;
            --accent-hover: #6366f1;
            --text-primary: #e2e8f0;
            --text-secondary: #94a3b8;
            --text-muted: #64748b;
            --border: #2d3748;
            --success: #10b981;
            --user-bubble: #3b82f6;
            --ai-bubble: #1e293b;
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: var(--bg-primary);
            color: var(--text-primary);
            min-height: 100vh;
            display: flex;
        }
        
        /* 侧边栏 */
        .sidebar {
            width: 280px;
            background: var(--bg-tertiary);
            border-right: 1px solid var(--border);
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
        }
        
        .sidebar-header {
            padding: 20px;
            border-bottom: 1px solid var(--border);
        }
        
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 18px;
            font-weight: 600;
        }
        
        .logo-icon {
            width: 36px;
            height: 36px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 20px;
        }
        
        .new-chat-btn {
            width: 100%;
            margin-top: 16px;
            padding: 12px;
            background: var(--accent);
            border: none;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            transition: background 0.2s;
        }
        
        .new-chat-btn:hover {
            background: var(--accent-hover);
        }
        
        .chat-history {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
        }
        
        .history-item {
            padding: 12px;
            border-radius: 8px;
            cursor: pointer;
            margin-bottom: 4px;
            color: var(--text-secondary);
            font-size: 14px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            transition: background 0.2s;
        }
        
        .history-item:hover, .history-item.active {
            background: var(--bg-secondary);
            color: var(--text-primary);
        }
        
        .sidebar-footer {
            padding: 16px;
            border-top: 1px solid var(--border);
        }
        
        .rag-status {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 13px;
            color: var(--text-secondary);
        }
        
        .status-dot {
            width: 8px;
            height: 8px;
            background: var(--success);
            border-radius: 50%;
        }
        
        /* 主聊天区域 */
        .main-content {
            flex: 1;
            display: flex;
            flex-direction: column;
            min-width: 0;
        }
        
        .chat-header {
            padding: 16px 24px;
            border-bottom: 1px solid var(--border);
            display: flex;
            align-items: center;
            justify-content: space-between;
        }
        
        .chat-title {
            font-size: 16px;
            font-weight: 500;
        }
        
        .toggle-rag {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        .toggle-switch {
            position: relative;
            width: 44px;
            height: 24px;
            background: var(--border);
            border-radius: 12px;
            cursor: pointer;
            transition: background 0.2s;
        }
        
        .toggle-switch.active {
            background: var(--accent);
        }
        
        .toggle-switch::after {
            content: '';
            position: absolute;
            width: 20px;
            height: 20px;
            background: white;
            border-radius: 50%;
            top: 2px;
            left: 2px;
            transition: transform 0.2s;
        }
        
        .toggle-switch.active::after {
            transform: translateX(20px);
        }
        
        /* 消息区域 */
        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 24px;
            display: flex;
            flex-direction: column;
            gap: 24px;
        }
        
        .message {
            display: flex;
            gap: 16px;
            max-width: 900px;
            margin: 0 auto;
            width: 100%;
        }
        
        .message-avatar {
            width: 36px;
            height: 36px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            justify-content: center;
            flex-shrink: 0;
            font-size: 16px;
        }
        
        .message.user .message-avatar {
            background: var(--user-bubble);
        }
        
        .message.assistant .message-avatar {
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
        }
        
        .message-content {
            flex: 1;
            min-width: 0;
        }
        
        .message-role {
            font-size: 13px;
            font-weight: 600;
            margin-bottom: 6px;
            color: var(--text-secondary);
        }
        
        .message-text {
            background: var(--ai-bubble);
            padding: 16px;
            border-radius: 12px;
            line-height: 1.7;
            font-size: 15px;
            white-space: pre-wrap;
            word-break: break-word;
        }
        
        .message.user .message-text {
            background: var(--user-bubble);
        }
        
        .message-sources {
            margin-top: 12px;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
        }
        
        .source-tag {
            background: rgba(79, 70, 229, 0.2);
            color: var(--accent-hover);
            padding: 4px 10px;
            border-radius: 6px;
            font-size: 12px;
            display: flex;
            align-items: center;
            gap: 4px;
        }
        
        /* 欢迎页面 */
        .welcome-screen {
            flex: 1;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            padding: 40px;
            text-align: center;
        }
        
        .welcome-icon {
            width: 80px;
            height: 80px;
            background: linear-gradient(135deg, var(--accent), #8b5cf6);
            border-radius: 20px;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 40px;
            margin-bottom: 24px;
        }
        
        .welcome-title {
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 12px;
        }
        
        .welcome-subtitle {
            color: var(--text-secondary);
            font-size: 16px;
            margin-bottom: 40px;
        }
        
        .quick-questions {
            display: grid;
            grid-template-columns: repeat(2, 1fr);
            gap: 12px;
            max-width: 700px;
        }
        
        .quick-question {
            background: var(--bg-secondary);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            text-align: left;
            cursor: pointer;
            transition: all 0.2s;
        }
        
        .quick-question:hover {
            border-color: var(--accent);
            background: rgba(79, 70, 229, 0.1);
        }
        
        .quick-question-icon {
            font-size: 20px;
            margin-bottom: 8px;
        }
        
        .quick-question-text {
            font-size: 14px;
            color: var(--text-secondary);
        }
        
        /* 输入区域 */
        .chat-input-area {
            padding: 20px 24px;
            border-top: 1px solid var(--border);
            background: var(--bg-secondary);
        }
        
        .input-container {
            max-width: 900px;
            margin: 0 auto;
            position: relative;
        }
        
        .chat-input {
            width: 100%;
            background: var(--bg-primary);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 16px 60px 16px 20px;
            color: var(--text-primary);
            font-size: 15px;
            resize: none;
            min-height: 56px;
            max-height: 200px;
            font-family: inherit;
            line-height: 1.5;
        }
        
        .chat-input:focus {
            outline: none;
            border-color: var(--accent);
        }
        
        .chat-input::placeholder {
            color: var(--text-muted);
        }
        
        .send-btn {
            position: absolute;
            right: 12px;
            bottom: 12px;
            width: 36px;
            height: 36px;
            background: var(--accent);
            border: none;
            border-radius: 10px;
            color: white;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            transition: background 0.2s;
        }
        
        .send-btn:hover:not(:disabled) {
            background: var(--accent-hover);
        }
        
        .send-btn:disabled {
            opacity: 0.5;
            cursor: not-allowed;
        }
        
        .input-hint {
            text-align: center;
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 8px;
        }
        
        /* 加载动画 */
        .typing-indicator {
            display: flex;
            gap: 4px;
            padding: 8px 0;
        }
        
        .typing-dot {
            width: 8px;
            height: 8px;
            background: var(--text-muted);
            border-radius: 50%;
            animation: bounce 1.4s infinite ease-in-out;
        }
        
        .typing-dot:nth-child(1) { animation-delay: -0.32s; }
        .typing-dot:nth-child(2) { animation-delay: -0.16s; }
        
        @keyframes bounce {
            0%, 80%, 100% { transform: scale(0); }
            40% { transform: scale(1); }
        }
        
        /* 响应式 */
        @media (max-width: 768px) {
            .sidebar {
                display: none;
            }
            
            .quick-questions {
                grid-template-columns: 1fr;
            }
            
            .chat-input-area {
                padding: 12px;
            }
        }
        
        /* 滚动条样式 */
        ::-webkit-scrollbar {
            width: 6px;
        }
        
        ::-webkit-scrollbar-track {
            background: transparent;
        }
        
        ::-webkit-scrollbar-thumb {
            background: var(--border);
            border-radius: 3px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: var(--text-muted);
        }
    </style>
</head>
<body>
    <!-- 侧边栏 -->
    <aside class="sidebar">
        <div class="sidebar-header">
            <div class="logo">
                <div class="logo-icon">⚖️</div>
                <span>智能法律助手</span>
            </div>
            <button class="new-chat-btn" onclick="newChat()">
                <span>+</span> 新对话
            </button>
        </div>
        <div class="chat-history" id="chatHistory">
            <div class="history-item active">📝 当前对话</div>
        </div>
        <div class="sidebar-footer">
            <div class="rag-status">
                <span class="status-dot"></span>
                <span>RAG知识库已就绪</span>
            </div>
        </div>
    </aside>

    <!-- 主内容区 -->
    <main class="main-content">
        <header class="chat-header">
            <div class="chat-title">法律智能问答 · RAG增强</div>
            <div class="toggle-rag">
                <span>知识库增强</span>
                <div class="toggle-switch active" id="ragToggle" onclick="toggleRag()"></div>
            </div>
        </header>

        <div class="chat-messages" id="chatMessages">
            <!-- 欢迎页面 -->
            <div class="welcome-screen" id="welcomeScreen">
                <div class="welcome-icon">⚖️</div>
                <h1 class="welcome-title">智能法律助手</h1>
                <p class="welcome-subtitle">基于RAG增强的专业法律问答系统，为您提供准确的法律咨询服务</p>
                <div class="quick-questions">
                    <div class="quick-question" onclick="askQuestion('公司拖欠工资两个月了，我应该怎么办？')">
                        <div class="quick-question-icon">💼</div>
                        <div class="quick-question-text">公司拖欠工资怎么维权？</div>
                    </div>
                    <div class="quick-question" onclick="askQuestion('网购买到假货可以要求几倍赔偿？')">
                        <div class="quick-question-icon">🛒</div>
                        <div class="quick-question-text">消费欺诈如何索赔？</div>
                    </div>
                    <div class="quick-question" onclick="askQuestion('夫妻离婚时房产如何分割？')">
                        <div class="quick-question-icon">🏠</div>
                        <div class="quick-question-text">离婚财产分割问题</div>
                    </div>
                    <div class="quick-question" onclick="askQuestion('发生交通事故后应该如何处理和索赔？')">
                        <div class="quick-question-icon">🚗</div>
                        <div class="quick-question-text">交通事故索赔流程</div>
                    </div>
                </div>
            </div>
        </div>

        <div class="chat-input-area">
            <div class="input-container">
                <textarea 
                    class="chat-input" 
                    id="chatInput" 
                    placeholder="请输入您的法律问题..."
                    rows="1"
                    onkeydown="handleKeyDown(event)"
                    oninput="autoResize(this)"
                ></textarea>
                <button class="send-btn" id="sendBtn" onclick="sendMessage()">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M22 2L11 13M22 2l-7 20-4-9-9-4 20-7z"/>
                    </svg>
                </button>
            </div>
            <div class="input-hint">支持跨设备调用 · API: POST /api/chat</div>
        </div>
    </main>

    <script>
        let useRag = true;
        let chatHistory = [];
        let isLoading = false;

        function toggleRag() {
            useRag = !useRag;
            document.getElementById('ragToggle').classList.toggle('active', useRag);
        }

        function newChat() {
            chatHistory = [];
            document.getElementById('chatMessages').innerHTML = document.getElementById('welcomeScreen').outerHTML;
            document.getElementById('chatInput').value = '';
        }

        function askQuestion(question) {
            document.getElementById('chatInput').value = question;
            sendMessage();
        }

        function autoResize(textarea) {
            textarea.style.height = 'auto';
            textarea.style.height = Math.min(textarea.scrollHeight, 200) + 'px';
        }

        function handleKeyDown(event) {
            if (event.key === 'Enter' && !event.shiftKey) {
                event.preventDefault();
                sendMessage();
            }
        }

        function addMessage(role, content, sources = []) {
            const welcomeScreen = document.getElementById('welcomeScreen');
            if (welcomeScreen) welcomeScreen.remove();

            const messagesDiv = document.getElementById('chatMessages');
            const isUser = role === 'user';
            
            const messageHtml = `
                <div class="message ${role}">
                    <div class="message-avatar">${isUser ? '👤' : '⚖️'}</div>
                    <div class="message-content">
                        <div class="message-role">${isUser ? '你' : '法律助手'}</div>
                        <div class="message-text">${content}</div>
                        ${sources.length > 0 ? `
                            <div class="message-sources">
                                ${sources.map(s => `<span class="source-tag">📚 ${s}</span>`).join('')}
                            </div>
                        ` : ''}
                    </div>
                </div>
            `;
            
            messagesDiv.insertAdjacentHTML('beforeend', messageHtml);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
            
            return messagesDiv.lastElementChild;
        }

        function addTypingIndicator() {
            const welcomeScreen = document.getElementById('welcomeScreen');
            if (welcomeScreen) welcomeScreen.remove();

            const messagesDiv = document.getElementById('chatMessages');
            const indicatorHtml = `
                <div class="message assistant" id="typingIndicator">
                    <div class="message-avatar">⚖️</div>
                    <div class="message-content">
                        <div class="message-role">法律助手</div>
                        <div class="message-text">
                            <div class="typing-indicator">
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                                <div class="typing-dot"></div>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            messagesDiv.insertAdjacentHTML('beforeend', indicatorHtml);
            messagesDiv.scrollTop = messagesDiv.scrollHeight;
        }

        function removeTypingIndicator() {
            const indicator = document.getElementById('typingIndicator');
            if (indicator) indicator.remove();
        }

        async function sendMessage() {
            if (isLoading) return;
            
            const input = document.getElementById('chatInput');
            const prompt = input.value.trim();
            if (!prompt) return;

            isLoading = true;
            document.getElementById('sendBtn').disabled = true;
            input.value = '';
            input.style.height = 'auto';

            // 添加用户消息
            addMessage('user', prompt);
            chatHistory.push({ role: 'user', content: prompt });

            // 显示加载状态
            addTypingIndicator();

            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        prompt: prompt,
                        use_rag: useRag,
                        stream: false,
                        history: chatHistory.slice(-6)
                    })
                });

                const data = await response.json();
                removeTypingIndicator();

                if (data.answer) {
                    addMessage('assistant', data.answer, data.sources || []);
                    chatHistory.push({ role: 'assistant', content: data.answer });
                } else {
                    addMessage('assistant', '抱歉，服务暂时不可用，请稍后重试。');
                }
            } catch (error) {
                removeTypingIndicator();
                addMessage('assistant', `请求失败: ${error.message}。请确保Ollama服务已启动。`);
            }

            isLoading = false;
            document.getElementById('sendBtn').disabled = false;
        }
    </script>
</body>
</html>
    """

# ================= 启动服务 =================
if __name__ == "__main__":
    print("🚀 启动智能法律助手服务...")
    print("📖 API文档: http://localhost:8000/docs")
    print("💬 聊天界面: http://localhost:8000")
    uvicorn.run(
        "llama_fastapi:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )
