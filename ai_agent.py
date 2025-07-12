import os
import requests
from dotenv import load_dotenv
from langchain.llms.base import LLM
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import Chroma
from typing import Any, List, Optional, Mapping

# 加载环境变量
load_dotenv()


class SiliconFlowLLM(LLM):
    """自定义硅基流动语言模型封装"""

    @property
    def _llm_type(self) -> str:
        return "siliconflow"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        model = os.getenv("MODEL_NAME", "SiliconFlow/GLM4-9B-Chat")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 1024,
            "temperature": 0.7
        }

        try:
            response = requests.post(
                "https://api.siliconflow.cn/v1/chat/completions",
                headers=headers,
                json=payload
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error: {str(e)}"


class SiliconFlowEmbeddings(Embeddings):
    """自定义硅基流动嵌入模型封装"""

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        api_key = os.getenv("SILICONFLOW_API_KEY")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-v1")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        embeddings = []
        for text in texts:
            payload = {"input": text, "model": model}
            try:
                response = requests.post(
                    "https://api.siliconflow.cn/v1/embeddings",
                    headers=headers,
                    json=payload
                )
                response.raise_for_status()
                embeddings.append(response.json()["data"][0]["embedding"])
            except:
                embeddings.append([0.0] * 768)  # 返回空嵌入作为后备

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]


class ProfessionalQAAgent:
    def __init__(self):
        # 初始化大模型
        self.llm = SiliconFlowLLM()

        # 初始化嵌入模型
        self.embeddings = SiliconFlowEmbeddings()

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
            result = qa_chain({"query": question})
            return {
                "answer": result["result"],
                "sources": [doc.metadata["source"] for doc in result["source_documents"]]
            }
        else:
            # 直接使用模型生成答案
            return {"answer": self.llm(question)}

    def cli_interaction(self):
        """命令行交互界面"""
        print("专业问答AI助手 (硅基流动版) | 输入'exit'退出")
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