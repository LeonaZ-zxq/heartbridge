"""设计说明页：把工程决策直接展示给访问者（也是给面试官看的）。"""
from __future__ import annotations

import streamlit as st
from shared import (api_key_sidebar, backend_name, backend_note, crisis_sidebar,
                    footer, get_cards, is_demo, page_setup)

page_setup("设计与隐私", "🔒")
api_key_sidebar()
crisis_sidebar()

st.title("🔒 这个系统是怎么设计的")

cards = get_cards()
c1, c2, c3, c4 = st.columns(4)
c1.metric("知识卡片", len(cards))
c2.metric("危机检测召回率", "100%",
          help="53 条评测集，14 条危机样本全部命中。作为 CI 发布门禁：漏一条则构建失败")
c3.metric("危机检测误报率", "0%",
          help="24 条普通消息，含中文「死」字惯用语等对抗样本")
c4.metric("检索 Recall@3", "70.0%",
          help="30 条改写式留出集查询，dense 后端，93 张卡；同一条件下 BM25 为 33.3%。"
               "注意：该留出集已被用于后端选择与索引设计，因此这个数字适合用来做后端间比较，"
               "不适合当作干净的泛化估计——见 docs/EVALUATION.md 第 4.4 节")

st.caption(f"当前检索后端：`{backend_name()}`"
           + (f"　—　{backend_note()}" if backend_note() else ""))

st.divider()
st.markdown("""
### 安全优先于功能

- **危机检测走规则优先、LLM 二级，而且是不对称的**：语言模型只被允许**升高**风险等级，永远不能降低。一个非确定性组件不应该有权关掉一个已经亮起的安全信号。
- **危机回复是人工审查过的固定模板，不是生成的**。在最高风险的时刻，可预测性和可审计性比个性化重要。相同输入永远得到相同输出。
- **危机分支在检索和生成之前短路**，所以危机时看到的内容与模型状态无关。

### 内容可溯源

- 每一条建议都能指回一张具体卡片，每张卡片都能指回 Beyond Blue 或 Healthdirect 的具体页面。
- **躯体化卡片在数据模型层面就要求权威来源**——没有来源的卡片在加载时直接报错，进不了知识库。
- 生成的每个回复选项必须引用一张**实际被检索到**的卡片；引用了没检索到的卡片会被判定为幻觉并丢弃。

### 隐私：本地优先

- 伴侣的心理健康信息是最高敏感级的个人数据，两条使用路径分别处理：
  **本机运行**时档案存在使用者自己机器的 SQLite（`data/heartbridge.db`），不进仓库、不上云；
  **这个公开实例**上则完全不落盘——档案只存在当前浏览器会话的服务端内存，刷新即消失，想留下来只能自己导出成 JSON 文件。
  这不是省事：公开实例是所有访问者共用的一个进程，写盘既是隐私问题也会让数据在访问者之间串。代码里**根本没有网页写盘那条路径**，并有测试钉住这一点。
- 向量检索用本地 CPU 模型，**使用者的文字不会发送给任何第三方**——这个隐私要求反过来决定了模型选型。
- 档案所有字段可选（数据最小化），并提供一键删除。
- **这个公开 demo 只使用虚构数据。**

### 评测

在一个独立撰写、未参与任何调优的留出集上报告结果，而不是报开发集上那个好看但被污染的数字。完整方法论见仓库的 `docs/EVALUATION.md`。
""")

st.link_button("在 GitHub 上查看源码与评测细节",
               "https://github.com/LeonaZ-zxq/heartbridge", use_container_width=True)
footer()
