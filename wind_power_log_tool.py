import streamlit as st
from openai import OpenAI
import json
import pandas as pd
from io import BytesIO

# ================= 1. 配置区域 (也就这里需要改) =================
# 使用Streamlit secrets管理API Key
# 在部署时，需要在Streamlit Cloud的Secrets设置中添加以下内容：
# [DEFAULT]
# MY_API_KEY = "你的API Key"
# MY_BASE_URL = "https://api.deepseek.com"

# 从secrets获取API Key和Base URL
# 处理secrets不存在的情况
try:
    MY_API_KEY = st.secrets["MY_API_KEY"]
    MY_BASE_URL = st.secrets["MY_BASE_URL"]
except Exception:
    # 如果secrets不存在，显示错误提示
    st.error("⚠️ 请先配置API Key！")
    st.info("""
    在本地运行时，请创建 `.streamlit/secrets.toml` 文件：
    ```
    MY_API_KEY = "你的API Key"
    MY_BASE_URL = "https://api.deepseek.com"
    ```
    
    在Streamlit Cloud部署时，请在App设置中添加Secrets。
    """)
    st.stop()

# 初始化连接器
client = OpenAI(api_key=MY_API_KEY, base_url=MY_BASE_URL)


# ================= 2. 定义大模型处理函数 (大脑) =================
def ai_process(text_input):
    """
    这个函数负责：把乱七八糟的文字 -> 发给AI -> 拿回整齐的JSON数据
    """
    # 这里的提示词(System Prompt)就是你控制AI输出格式的遥控器
    # 我们强制要求它输出 JSON 格式
    prompt = """
    你是一个专业的风电运维日志分析助手，负责从原始日志中提取关键信息并转换为结构化数据。
    
    提取规则：
    1. 必须提取以下字段：
       - 机位：风机的编号或位置，如"A01"、"B12"等
       - 工单号：维修或维护任务的编号
       - 日期：工作执行的日期，格式如"2026年2月10日"
       - 缺陷描述/工作内容：详细描述故障或工作任务
       - 解决措施/工作完成情况：详细描述采取的解决措施或完成的工作内容
       - 问题处理人员/调试人员：参与工作的人员名单，包括所有相关人员
       - 遗留问题：未解决或需要后续处理的问题
    
    2. 处理规则：
       - 如果某个字段没有相关信息，填"未提及"
       - 完整提取"今日工作完成情况"的内容，整体放置于"解决措施/工作完成情况"内
       - 完整提取"工作班成员："的内容，整体放置于"问题处理人员/调试人员"内
       - 保持原始信息的完整性，不要遗漏任何重要细节
       - 对于日期格式，统一转换为"年-月-日"格式，如"2026-02-10"
    
    3. 输入格式处理：
       - 优先处理以下模板格式的输入：
         时间：
         工单号：
         工作内容：
         工作班成员：
         今日工作完成情况:
         遗留问题：
       - 对于这种模板格式，将"工作内容"对应到"缺陷描述/工作内容"字段
       - 将"今日工作完成情况"对应到"解决措施/工作完成情况"字段
       - 将"工作班成员"对应到"问题处理人员/调试人员"字段
    
    4. 输出要求：
       - 严禁输出任何解释性文字，只输出JSON数据
       - 必须严格按照JSON格式输出，使用列表形式
       - 确保JSON格式正确，无语法错误
       - 字段名称必须与要求完全一致
    
    输入示例1（模板格式）：
    时间：2026-03-01
    工单号：205450
    工作内容：A06机组季度巡检
    工作班成员：海装：王学兵、王磊
    今日工作完成情况: 1、业主ERP系统巡检（已完成） 2、放置塔基应急药品（已完成） 3、机舱盐雾监测系统报警消缺（重启机舱盐雾监测系统后恢复正常） 4、轮毂散热风扇螺栓排查（无异常）
    遗留问题：主轴跑圈标记拍照。
    
    输入示例2（紧凑格式）：
    A06 	 205450 	 2026-03-01 	 A06机组季度巡检 	 1、业主ERP系统巡检（已完成） 2、放置塔基应急药品（已完成） 3、机舱盐雾监测系统报警消缺（重启机舱盐雾监测系统后恢复正常） 4、轮毂散热风扇螺栓排查（无异常） 	 海装：王学兵、王磊 	 主轴跑圈标记拍照。
    
    输出示例：
    [
        {
            "机位": "A06",
            "工单号": "205450",
            "日期": "2026-03-01",
            "缺陷描述/工作内容": "A06机组季度巡检",
            "解决措施/工作完成情况": "1、业主ERP系统巡检（已完成） 2、放置塔基应急药品（已完成） 3、机舱盐雾监测系统报警消缺（重启机舱盐雾监测系统后恢复正常） 4、轮毂散热风扇螺栓排查（无异常）",
            "问题处理人员/调试人员": "海装：王学兵、王磊",
            "遗留问题": "主轴跑圈标记拍照。"
        }
    ]
    """

    try:
        # 输入验证
        if not text_input or len(text_input.strip()) < 10:
            return "{\"error\": \"输入文本太短，请提供完整的日志信息\"}"
        
        response = client.chat.completions.create(
            model="deepseek-chat",  # 模型名字
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text_input}
            ],
            response_format={'type': 'json_object'},  # 关键！强制变乖，只吐数据
            temperature=0.1  # 让AI严谨点，别瞎发挥
        )
        return response.choices[0].message.content
    except Exception as e:
        # 详细的错误处理
        error_message = f"出错了: {str(e)}"
        st.error(f"API调用失败: {str(e)}")
        return f"{{\"error\": \"{error_message}\"}}"


# ================= 3. 搭建网页界面 (Streamlit) =================
st.set_page_config(page_title="风电日志清洗神器", layout="wide")

# 添加页面标题和说明
st.title("🌪️ 风电运维日志 -> 结构化表格工具")
st.markdown("### 功能介绍")
st.markdown("本工具可以帮助您将杂乱的风电运维日志自动转换为结构化表格，方便您整理和分析数据。")
st.markdown("### 使用方法")
st.markdown("1. 在左侧文本框中粘贴您的风电运维日志")
st.markdown("2. 点击'开始清洗数据'按钮")
st.markdown("3. 在右侧查看结构化表格结果")
st.markdown("4. 下载Excel文件保存结果")

# 创建两列布局：左边输入，右边输出
col1, col2 = st.columns(2)

with col1:
    st.header("1. 输入原始日志")
    # 创建一个大文本框
    user_text = st.text_area("请粘贴文本:", height=400,
                             placeholder="例如：A12机位昨天报了变桨故障，张伟去修了3个小时搞好了...")
    
    # 添加示例输入按钮
    if st.button("加载示例日志"):
        sample_log = """A01机位 工单号：12345 日期：2026年2月10日
缺陷描述：变桨轴承润滑系统故障
解决措施：1、上机排查相关线路未发现异常，进一步排查发现为变桨轴承润滑泵内部存在空气，已将空气排出，故障消除。运行润滑泵一小时，未发现异常。
2、排查3个面轮毂风扇固定螺栓，未发现异常
工作班成员：海装：黄啟洪、杨晓君、丁文
遗留问题：无"""
        user_text = sample_log

    # 创建一个按钮
    run_btn = st.button("开始清洗数据 🚀", type="primary")

with col2:
    st.header("2. 结果展示")

    if run_btn:  # 如果按钮被点击了
        if not user_text:
            st.warning("请先在左边输入点东西！")
        else:
            # 添加加载动画
            with st.spinner("正在生成..."):
                # 1. 调用大脑函数
                raw_json = ai_process(user_text)

                # 2. 尝试把 AI 返回的 JSON 转换成表格
                try:
                    # 把字符串变成数据
                    data_obj = json.loads(raw_json)

                    # 这里的逻辑是兼容 DeepSeek 可能包裹在 key 里的情况
                    if isinstance(data_obj, dict):
                        # 如果AI很啰嗦给了一个 {"results": [...]}, 我们取这个列表
                        key = list(data_obj.keys())[0]
                        data_list = data_obj[key]
                    else:
                        data_list = data_obj

                    # 变成表格
                    df = pd.DataFrame(data_list)

                    # 展示表格
                    st.dataframe(df, use_container_width=True)

                    # 添加一键复制功能（使用JavaScript实现剪贴板复制）
                    import base64
                    
                    # 将DataFrame转换为制表符分隔的文本
                    def to_clipboard_text(df):
                        return df.to_csv(index=False, sep='\t')
                    
                    clipboard_text = to_clipboard_text(df)
                    
                    # 创建JavaScript复制功能
                    b64 = base64.b64encode(clipboard_text.encode()).decode()
                    href = f'<a href="#" onclick="navigator.clipboard.writeText(atob(\'{b64}\')); document.getElementById(\'copy-notification\').style.display = \'block\'; setTimeout(function() {{ document.getElementById(\'copy-notification\').style.opacity = \'0\'; }}, 2000); return false;">一键复制到剪贴板 📋</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    
                    # 添加复制成功提示
                    st.markdown('<div id="copy-notification" style="background-color: #4CAF50; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; display: none; transition: opacity 1s ease-in-out;">已完成复制！</div>', unsafe_allow_html=True)

                    # 3. 甚至可以给一个下载 Excel 的按钮
                    # (需要在内存里转换一下，这行稍微复杂，可以先忽略原理)
                    # 只要知道这给了你一个下载 Excel 的功能
                    def to_excel(df):
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            df.to_excel(writer, index=False, sheet_name='风电运维日志')
                        processed_data = output.getvalue()
                        return processed_data

                    excel_data = to_excel(df)
                    st.download_button(
                        label="下载 Excel 文件 📊",
                        data=excel_data,
                        file_name="风电运维日志.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
                    st.success("处理完成！你可以复制数据或下载 Excel 了 👇")
                except Exception as e:
                    st.error(f"转换失败，可能是 AI 返回格式不对。错误信息：{e}")
                    st.text("AI原始返回内容：")
                    st.code(raw_json)
    else:
        st.info("请在左侧输入日志并点击'开始清洗数据'按钮")