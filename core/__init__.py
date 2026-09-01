"""HeartBridge — 抑郁症伴侣沟通支持 agent 的核心逻辑包。

分层：knowledge(知识库+检索) / safety(危机检测) / engine(回复生成) / profile / care。
所有前端（CLI、Discord bot、Streamlit）都只是这个包的薄调用层。
"""

__version__ = "0.1.0"
