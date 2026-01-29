import feedparser
import urllib.parse
import sys

# --- 配置区域参数 ---
REGIONS = {
    '1': {
        'name': '🇨🇳 中国 (中文)',
        'params': '&hl=zh-CN&gl=CN&ceid=CN:zh-CN'
    },
    '2': {
        'name': '🇺🇸 美国 (英文)',
        'params': '&hl=en-US&gl=US&ceid=US:en'
    }
}

def fetch_rss(query, region_key):
    """
    内部函数：根据指定的区域代码抓取新闻
    """
    config = REGIONS[region_key]
    print(f"\n--- 正在获取 {config['name']} 消息 ---")
    
    encoded_query = urllib.parse.quote(query)
    # 基础 URL + 动态的区域参数
    rss_url = f"https://news.google.com/rss/search?q={encoded_query}+when:1d{config['params']}"
    
    try:
        feed = feedparser.parse(rss_url)
    except Exception as e:
        print(f" 连接超时或错误: {e}")
        return

    if not feed.entries:
        print(f"在 {config['name']} 未找到相关消息。")
        return

    # 打印前 5 条 (如果选Both，为了防止刷屏，每种语言只显示5条)
    for i, entry in enumerate(feed.entries[:5], 1):
        print(f"{i}. {entry.title}")
        print(f"   发布: {entry.published} | 来源: {entry.source.title}")
        print(f"   链接: {entry.link}")
        print("-" * 30)

def get_financial_news(query, mode):
    """
    主逻辑：根据用户模式调用不同的 RSS 源
    """
    print(f"\n 正在搜集关于【{query}】的最新情报...")
    
    if mode == '1':
        fetch_rss(query, '1') # 只搜中文
    elif mode == '2':
        fetch_rss(query, '2') # 只搜英文
    elif mode == '3':
        fetch_rss(query, '1') # 先搜中文
        fetch_rss(query, '2') # 再搜英文

# --- 主程序入口 ---
if __name__ == "__main__":
    print("欢迎使用财经新闻聚合器 2.0")
    print("请选择搜索区域/语言：")
    print("1. 🇨🇳 仅中国 (中文)")
    print("2. 🇺🇸 仅美国 (英文)")
    print("3.  混合模式 (Both - 中文+英文)")
    
    # 1. 设定模式
    while True:
        mode = input("请输入模式编号 (1/2/3): ").strip()
        if mode in ['1', '2', '3']:
            break
        print("输入无效，请输入 1, 2 或 3")
    
    mode_name = "混合模式" if mode == '3' else REGIONS[mode]['name']
    print(f"\n 已设定为: {mode_name}")

    # 2. 开始循环搜索
    while True:
        user_input = input("\n请输入股票名称或大宗商品（输入 q 退出）: ")
        if user_input.lower() == 'q':
            print(" 程序已退出")
            break
        
        if not user_input.strip():
            continue
            
        get_financial_news(user_input, mode)