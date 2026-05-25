import streamlit as st
import requests
import uuid
import os
import json
from datetime import datetime
from typing import List, Dict, Any

st.set_page_config(
    page_title="文档对话机器人 - RAG系统",
    page_icon="🤖",
    layout="wide"
)

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    .main-header {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
        padding: 20px;
        border-radius: 16px;
        margin-bottom: 20px;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.08);
    }

    .glow-text {
        text-shadow: 0 0 20px rgba(99, 102, 241, 0.5);
        color: white;
    }

    .glass-card {
        background: rgba(255, 255, 255, 0.03);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border-radius: 16px;
        padding: 20px;
        transition: all 0.3s ease;
    }

    .glass-card:hover {
        background: rgba(255, 255, 255, 0.05);
        border-color: rgba(99, 102, 241, 0.3);
        transform: translateY(-2px);
    }

    .neon-border {
        position: relative;
        border-radius: 16px;
    }

    .neon-border::before {
        content: '';
        position: absolute;
        inset: -2px;
        background: linear-gradient(45deg, #6366f1, #8b5cf6, #06b6d4, #6366f1);
        background-size: 400% 400%;
        border-radius: inherit;
        z-index: -1;
        animation: gradient-rotate 8s ease infinite;
        opacity: 0.5;
    }

    @keyframes gradient-rotate {
        0%, 100% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
    }

    .status-online {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    .status-offline {
        background: linear-gradient(135deg, #f093fb, #f5576c);
        color: white;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
    }

    .pulse-dot {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; transform: scale(1); }
        50% { opacity: 0.5; transform: scale(1.2); }
    }

    .message-user {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        border-radius: 16px 16px 4px 16px;
        padding: 16px;
        color: white;
    }

    .message-assistant {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px 16px 16px 4px;
        padding: 16px;
    }

    .typing-indicator span {
        animation: typing 1.4s infinite both;
        display: inline-block;
        width: 8px;
        height: 8px;
        background: rgba(99, 102, 241, 0.8);
        border-radius: 50%;
        margin: 0 2px;
    }

    .typing-indicator span:nth-child(2) { animation-delay: 0.2s; }
    .typing-indicator span:nth-child(3) { animation-delay: 0.4s; }

    @keyframes typing {
        0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
        30% { transform: translateY(-10px); opacity: 1; }
    }

    .btn-primary {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
        border: none;
        color: white;
        padding: 12px 24px;
        border-radius: 12px;
        font-weight: 600;
        cursor: pointer;
    }

    .btn-primary:hover {
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.5);
        transform: translateY(-2px);
    }

    .btn-secondary {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        color: white;
        padding: 10px 20px;
        border-radius: 12px;
        cursor: pointer;
    }

    .btn-secondary:hover {
        background: rgba(255, 255, 255, 0.1);
        border-color: rgba(99, 102, 241, 0.5);
    }

    .stat-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(139, 92, 246, 0.1) 100%);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }

    .progress-bar {
        background: linear-gradient(90deg, #6366f1, #8b5cf6, #06b6d4);
        background-size: 200% 100%;
        animation: progress-shine 2s linear infinite;
        height: 6px;
        border-radius: 3px;
    }

    @keyframes progress-shine {
        0% { background-position: 200% 0; }
        100% { background-position: -200% 0; }
    }

    .source-card {
        background: rgba(99, 102, 241, 0.1);
        border-left: 3px solid #6366f1;
        padding: 12px;
        border-radius: 0 8px 8px 0;
        margin: 8px 0;
    }

    .input-dark {
        background: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: all 0.3s ease;
        border-radius: 12px;
        padding: 12px;
        color: white;
    }

    .input-dark:focus {
        border-color: #6366f1;
        box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2);
        outline: none;
    }

    .scrollbar-custom::-webkit-scrollbar {
        width: 6px;
    }

    .scrollbar-custom::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 3px;
    }

    .scrollbar-custom::-webkit-scrollbar-thumb {
        background: rgba(99, 102, 241, 0.5);
        border-radius: 3px;
    }

    .scrollbar-custom::-webkit-scrollbar-thumb:hover {
        background: rgba(99, 102, 241, 0.7);
    }

    .tab-active {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        border-radius: 12px;
        padding: 12px 24px;
    }

    .file-item {
        animation: slideIn 0.3s ease;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 8px;
        padding: 12px;
        margin: 8px 0;
    }

    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }

    .fade-in {
        animation: fadeIn 0.5s ease;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .stApp {
        background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    }

    .css-1d391kg, .css-1woncv, .css-1v0z1l8 {
        background: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_loaded" not in st.session_state:
        st.session_state.documents_loaded = False
    if "stats" not in st.session_state:
        st.session_state.stats = {"questions": 0, "responses": 0, "total_time": 0, "avg_time": 0}
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "current_model" not in st.session_state:
        st.session_state.current_model = "deepseek-chat"
    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = "similarity"
    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7
    if "top_k" not in st.session_state:
        st.session_state.top_k = 4

def check_health() -> bool:
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def upload_documents(files: List[Any]) -> Dict[str, Any]:
    if not files:
        return {"success": False, "message": "未选择文件"}

    try:
        files_data = []
        for file in files:
            files_data.append(("files", (file.name, file.getvalue(), file.type)))

        response = requests.post(
            f"{API_BASE_URL}/upload",
            files=files_data,
            timeout=60
        )

        if response.status_code == 200:
            result = response.json()
            if result.get("success"):
                st.session_state.documents_loaded = True
                for file in files:
                    st.session_state.uploaded_files.append(file.name)
            return result
        else:
            return {"success": False, "message": response.text}

    except Exception as e:
        return {"success": False, "message": str(e)}

def send_message(question: str, temperature: float = 0.7, top_k: int = 4) -> Dict[str, Any]:
    try:
        payload = {
            "question": question,
            "session_id": st.session_state.session_id,
            "temperature": temperature,
            "top_k": top_k
        }

        response = requests.post(
            f"{API_BASE_URL}/chat",
            json=payload,
            timeout=120
        )

        if response.status_code == 200:
            return response.json()
        else:
            return {"answer": f"错误: {response.text}", "sources": [], "time": 0}

    except Exception as e:
        return {"answer": f"连接错误: {str(e)}", "sources": [], "time": 0}

def export_chat():
    if not st.session_state.messages:
        st.warning("没有对话记录可导出")
        return

    chat_data = {
        "session_id": st.session_state.session_id,
        "export_time": datetime.now().isoformat(),
        "model": st.session_state.current_model,
        "messages": st.session_state.messages,
        "stats": st.session_state.stats
    }

    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="📥 导出JSON",
            data=json.dumps(chat_data, ensure_ascii=False, indent=2),
            file_name=f"对话记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )

    chat_text = f"文档对话机器人 - 对话记录\n"
    chat_text += "=" * 60 + "\n"
    chat_text += f"会话ID: {st.session_state.session_id}\n"
    chat_text += f"导出时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
    chat_text += f"模型: {st.session_state.current_model}\n"
    chat_text += "=" * 60 + "\n\n"

    for msg in st.session_state.messages:
        role = "👤 用户" if msg["role"] == "user" else "🤖 AI"
        chat_text += f"{role}:\n{msg['content']}\n"
        if msg.get("sources"):
            chat_text += "📑 引用来源:\n"
            for i, source in enumerate(msg["sources"], 1):
                chat_text += f"  [{i}] {source.get('source', '未知来源')}\n"
        chat_text += "\n" + "-" * 60 + "\n\n"

    with col2:
        st.download_button(
            label="📄 导出TXT",
            data=chat_text,
            file_name=f"对话记录_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True
        )

def clear_vector_store():
    try:
        response = requests.delete(f"{API_BASE_URL}/clear", timeout=10)
        if response.status_code == 200:
            st.session_state.documents_loaded = False
            st.session_state.uploaded_files = []
            return True
        return False
    except Exception as e:
        st.error(f"清除失败: {str(e)}")
        return False

def get_model_info():
    try:
        response = requests.get(f"{API_BASE_URL}/model_info", timeout=5)
        if response.status_code == 200:
            return response.json()
        return {"model": st.session_state.current_model}
    except:
        return {"model": st.session_state.current_model}

def main():
    init_session_state()

    session_id_short = st.session_state.session_id[:8]

    st.markdown(f"""
        <div class="main-header">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <div style="display: flex; align-items: center; gap: 16px;">
                    <div style="width: 48px; height: 48px; border-radius: 12px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); display: flex; align-items: center; justify-content: center;">
                        <span style="font-size: 24px;">🤖</span>
                    </div>
                    <div>
                        <h1 class="glow-text" style="font-size: 24px; margin-bottom: 4px;">🤖 文档对话机器人</h1>
                        <p style="color: rgba(255,255,255,0.6); font-size: 13px;">基于RAG技术的智能文档问答系统</p>
                    </div>
                </div>
                <div style="display: flex; align-items: center; gap: 12px;">
                    <span style="font-size: 13px; color: rgba(255,255,255,0.6);">会话: <code style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 4px;">{session_id_short}</code></span>
                    <span style="font-size: 13px; color: rgba(255,255,255,0.6);">模型: <code style="background: rgba(255,255,255,0.1); padding: 2px 8px; border-radius: 4px;">deepseek-chat</code></span>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    status = "在线" if check_health() else "离线"
    status_class = "status-online" if check_health() else "status-offline"

    col1, col2, col3, col4 = st.columns([1, 1, 1, 6])

    with col1:
        st.markdown(f"""
            <div class="glass-card" style="padding: 12px; text-align: center;">
                <div style="font-size: 24px; color: #6366f1; font-weight: bold;">{st.session_state.stats['questions']}</div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.6);">问题数</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div class="glass-card" style="padding: 12px; text-align: center;">
                <div style="font-size: 24px; color: #11998e; font-weight: bold;">{st.session_state.stats['responses']}</div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.6);">响应数</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        avg_time = st.session_state.stats['avg_time']
        st.markdown(f"""
            <div class="glass-card" style="padding: 12px; text-align: center;">
                <div style="font-size: 24px; color: #f5576c; font-weight: bold;">{avg_time:.1f}s</div>
                <div style="font-size: 12px; color: rgba(255,255,255,0.6);">平均响应</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
            <div style="text-align: right; padding: 12px;">
                <span class="{status_class}">● {status}</span>
            </div>
        """, unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["💬 对话", "📁 文档管理", "⚙️ 参数设置"])

    with tab1:
        chat_col, info_col = st.columns([3, 1])

        with chat_col:
            st.markdown('<div class="glass-card neon-border" style="height: 500px; display: flex; flex-direction: column;">', unsafe_allow_html=True)

            chat_container = st.container(height=400)

            with chat_container:
                for message in st.session_state.messages:
                    if message["role"] == "user":
                        st.markdown(f"""
                            <div style="display: flex; justify-content: flex-end; margin: 16px 0;">
                                <div class="message-user" style="max-width: 80%;">
                                    <div style="margin-bottom: 8px;">👤 <strong>您</strong></div>
                                    <div>{message['content']}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                            <div style="display: flex; justify-content: flex-start; margin: 16px 0;">
                                <div class="message-assistant" style="max-width: 80%;">
                                    <div style="margin-bottom: 8px;">🤖 <strong>AI助手</strong></div>
                                    <div>{message['content']}</div>
                                </div>
                            </div>
                        """, unsafe_allow_html=True)

                        if message.get("sources") and len(message["sources"]) > 0:
                            with st.expander("📑 查看引用来源", expanded=False):
                                for i, source in enumerate(message["sources"], 1):
                                    st.markdown(f"""
                                        <div class="source-card">
                                            <strong>[{i}] {source.get('source', '未知文档')}</strong>
                                            <p style="font-size: 12px; color: rgba(255,255,255,0.6); margin-top: 4px;">
                                                页码: {source.get('page', 'N/A')}
                                            </p>
                                            <p style="margin-top: 8px; font-size: 13px;">
                                                {source['content'][:300]}{'...' if len(source['content']) > 300 else ''}
                                            </p>
                                        </div>
                                    """, unsafe_allow_html=True)

                        if message.get("time"):
                            st.caption(f"⏱️ 响应时间: {message['time']:.2f}秒")

            user_input = st.text_area("输入您的问题...", height=80, placeholder="请输入您的问题，AI将基于上传的文档进行回答...", key="chat_input")

            col_send, col_clear = st.columns([3, 1])
            with col_send:
                if st.button("🚀 发送问题", use_container_width=True):
                    if user_input.strip():
                        st.session_state.messages.append({
                            "role": "user",
                            "content": user_input.strip(),
                            "timestamp": datetime.now().isoformat()
                        })
                        st.session_state.stats["questions"] += 1

                        with st.spinner("🤔 AI正在思考..."):
                            result = send_message(
                                user_input.strip(),
                                temperature=st.session_state.temperature,
                                top_k=st.session_state.top_k
                            )

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result.get("answer", "未收到响应"),
                            "sources": result.get("sources", []),
                            "time": result.get("time", 0),
                            "timestamp": datetime.now().isoformat()
                        })
                        st.session_state.stats["responses"] += 1
                        st.session_state.stats["total_time"] += result.get("time", 0)
                        if st.session_state.stats["responses"] > 0:
                            st.session_state.stats["avg_time"] = st.session_state.stats["total_time"] / st.session_state.stats["responses"]

                        st.rerun()

            with col_clear:
                if st.button("🗑️ 清空", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.stats = {"questions": 0, "responses": 0, "total_time": 0, "avg_time": 0}
                    st.rerun()

            st.markdown('</div>', unsafe_allow_html=True)

        with info_col:
            st.subheader("💡 推荐问题")
            recommended = [
                "这份文档的主要内容是什么？",
                "请总结文档的关键要点",
                "文档中有什么重要结论？",
                "文档提到了哪些解决方案？"
            ]

            for question in recommended:
                if st.button(question, use_container_width=True, key=f"rec_{question}"):
                    st.session_state.messages.append({
                        "role": "user",
                        "content": question,
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.stats["questions"] += 1

                    with st.spinner("🤔 AI正在思考..."):
                        result = send_message(
                            question,
                            temperature=st.session_state.temperature,
                            top_k=st.session_state.top_k
                        )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result.get("answer", "未收到响应"),
                        "sources": result.get("sources", []),
                        "time": result.get("time", 0),
                        "timestamp": datetime.now().isoformat()
                    })
                    st.session_state.stats["responses"] += 1
                    st.rerun()

            st.divider()

            st.subheader("🔗 快速操作")
            export_chat()

    with tab2:
        st.subheader("📁 文档管理")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            "选择文档文件",
            type=["pdf", "txt", "md", "docx"],
            accept_multiple_files=True,
            help="支持PDF、TXT、MD、DOCX格式"
        )

        if uploaded_files:
            if st.button("🚀 上传并处理", use_container_width=True):
                with st.spinner("正在处理文档..."):
                    result = upload_documents(uploaded_files)
                    if result.get("success"):
                        st.success(f"✅ 上传成功！已处理 {result.get('documents_processed', 0)} 个文档片段")
                    else:
                        st.error(f"❌ {result.get('message', '上传失败')}")

        if st.session_state.uploaded_files:
            st.subheader("已上传的文档")
            for idx, file in enumerate(st.session_state.uploaded_files, 1):
                st.markdown(f"""
                    <div class="file-item">
                        📄 {file}
                    </div>
                """, unsafe_allow_html=True)

            if st.button("🗑️ 清除所有文档", use_container_width=True):
                if clear_vector_store():
                    st.success("✅ 已清除所有文档")
                    st.rerun()
                else:
                    st.error("❌ 清除失败")

        st.markdown('</div>', unsafe_allow_html=True)

        st.divider()

        st.subheader("📖 使用说明")
        st.markdown("""
        1. **上传文档**: 在上方选择您的文档文件
        2. **点击上传**: 等待文档处理完成
        3. **开始提问**: 切换到对话标签页开始提问
        4. **查看来源**: 回答会显示引用的文档来源

        **支持的文件格式**: PDF、TXT、MD、DOCX
        """)

    with tab3:
        st.subheader("⚙️ 参数设置")

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)

        st.markdown("### 🧠 模型参数")
        temperature = st.slider(
            "温度参数",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.temperature,
            step=0.05,
            help="较低的值更精确，较高的值更有创意"
        )
        st.session_state.temperature = temperature

        st.markdown("### 🔍 检索参数")
        top_k = st.slider(
            "检索文档数量 (Top K)",
            min_value=1,
            max_value=10,
            value=st.session_state.top_k,
            step=1,
            help="从向量数据库中检索的相关文档数量"
        )
        st.session_state.top_k = top_k

        retrieval_mode = st.selectbox(
            "检索模式",
            ["similarity", "mmr"],
            index=0 if st.session_state.retrieval_mode == "similarity" else 1,
            help="similarity: 相似度检索 | mmr: 最大边际相关性"
        )
        st.session_state.retrieval_mode = retrieval_mode

        st.divider()

        st.markdown("### 📤 API配置")
        st.markdown(f"当前API地址: `{API_BASE_URL}`")

        if st.button("🔄 测试连接", use_container_width=True):
            if check_health():
                st.success("✅ 连接成功！")
            else:
                st.error("❌ 连接失败，请检查后端服务")

        st.divider()

        st.subheader("📝 系统信息")
        st.markdown(f"""
        - **会话ID**: `{st.session_state.session_id}`
        - **当前模型**: `{st.session_state.current_model}`
        - **文档数量**: {len(st.session_state.uploaded_files)}
        - **对话轮数**: {len(st.session_state.messages)}
        """)

        st.markdown('</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
