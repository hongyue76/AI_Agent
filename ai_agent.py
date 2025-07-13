import os
import requests
from dotenv import load_dotenv
from langchain_core.language_models import LLM
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import TextLoader
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma  # 修正导入路径
from typing import Any, List, Optional, Mapping

# 加载环境变量
load_dotenv()


class DeepSeekLLM(LLM):
    """DeepSeek语言模型封装"""

    @property
    def _llm_type(self) -> str:
        return "deepseek"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            return "错误: 未找到DEEPSEEK_API_KEY环境变量"

        model = os.getenv("MODEL_NAME", "deepseek-chat")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 512,
            "temperature": 0.7,
            "top_p": 0.9,
            "stream": False
        }

        try:
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if response.status_code == 200:
                return response.json()["choices"][0]["message"]["content"]
            else:
                return f"API错误: {response.status_code} - {response.text}"

        except Exception as e:
            return f"请求失败: {str(e)}"


class DeepSeekEmbeddings(Embeddings):
    """DeepSeek嵌入模型封装"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

        if not api_key:
            return [[0.0] * 1536 for _ in texts]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        embeddings = []
        for text in texts:
            payload = {"input": text, "model": model}
            try:
                response = requests.post(
                    "https://api.deepseek.com/v1/embeddings",
                    headers=headers,
                    json=payload,
                    timeout=15
                )
                if response.status_code == 200:
                    embeddings.append(response.json()["data"][0]["embedding"])
                else:
                    print(f"嵌入错误: {response.status_code} - {response.text}")
                    embeddings.append([0.0] * 1536)
            except Exception as e:
                print(f"请求失败: {str(e)}")
                embeddings.append([0.0] * 1536)

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class ProfessionalQAAgent:
    def __init__(self):
        # 初始化大模型
        self.llm = DeepSeekLLM()

        # 初始化嵌入模型
        self.embeddings = DeepSeekEmbeddings()

        # 知识库初始化
        self.knowledge_base = None

    def _initialize_knowledge_base(self, file_path):
        """构建知识库向量存储"""
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load_and_split()

        self.knowledge_base = Chroma.from_documents(
            documents,
            self.embeddings
        ).as_retriever()

    def answer_question(self, question: str, knowledge_source: str = None):
        """回答专业问题"""
        if knowledge_source and not self.knowledge_base:
            self._initialize_knowledge_base(knowledge_source)

        if self.knowledge_base:
            qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.knowledge_base,
                return_source_documents=True
            )
            result = qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.metadata["source"] for doc in result["source_documents"]]
            }
        else:
            return {"answer": self.llm.invoke(question)}

    def cli_interaction(self):
        """命令行交互界面"""
        print("专业问答AI助手 (DeepSeek版) | 输入'exit'退出")
        while True:
            query = input("\n问题: ")
            if query.lower() == 'exit':
                break
            response = self.answer_question(query)
            print(f"\n答案: {response['answer']}")
            if 'sources' in response and response['sources']:
                print(f"来源: {', '.join(response['sources'])}")


if __name__ == "__main__":
    agent = ProfessionalQAAgent()
    agent.cli_interaction()