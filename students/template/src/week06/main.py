import pandas as pd
import numpy as np
from pathlib import Path

def generate_marketing_data(file_path: Path):
    """
    生成 Q3 营销数据集。
    包含两个市场（NA, EU），各 500 条数据。
    """
    # 设定全局随机种子，确保每次生成的数据完全一致
    rng = np.random.default_rng(seed=2026)
    
    # ==========================================
    # 1. 模拟北美市场 (North America - NA)
    # 埋点规则：
    # - TV 和 Radio 效果完全相同 (均为 3.5)
    # - Social Media 毫无卵用 (0.0)
    # ==========================================
    n_na = 500
    tv_na = rng.uniform(50, 300, n_na)
    radio_na = rng.uniform(10, 100, n_na)
    social_na = rng.uniform(10, 150, n_na)
    holiday_na = rng.binomial(1, 0.2, n_na) # 20% 的时间是节假日
    
    # 真实的 DGP (Data Generating Process)
    noise_na = rng.normal(0, 15.0, n_na) # 噪音较大
    sales_na = (
        50.0                   # Intercept
        + 3.5 * tv_na          # TV
        + 3.5 * radio_na       # Radio (与 TV 完全相等)
        + 0.0 * social_na      # Social Media (无效参数！)
        + 25.0 * holiday_na    # 节假日效应
        + noise_na
    )
    
    df_na = pd.DataFrame({
        "Region": "NA",
        "TV_Budget": tv_na,
        "Radio_Budget": radio_na,
        "SocialMedia_Budget": social_na,
        "Is_Holiday": holiday_na,
        "Sales": sales_na
    })

    # ==========================================
    # 2. 模拟欧洲市场 (Europe - EU)
    # 埋点规则：
    # - Radio 的效果 (4.8) 远大于 TV (1.5)
    # - Social Media 是有效的 (1.2)
    # ==========================================
    n_eu = 500
    tv_eu = rng.uniform(20, 200, n_eu)
    radio_eu = rng.uniform(20, 150, n_eu)
    social_eu = rng.uniform(50, 250, n_eu) # 欧洲区社交媒体预算更高
    holiday_eu = rng.binomial(1, 0.25, n_eu) 
    
    noise_eu = rng.normal(0, 10.0, n_eu) # 噪音较小
    sales_eu = (
        30.0                   # Intercept
        + 1.5 * tv_eu          # TV (效果较弱)
        + 4.8 * radio_eu       # Radio (效果极强！)
        + 1.2 * social_eu      # Social Media (有微弱拉动效果)
        + 18.0 * holiday_eu    # 节假日效应
        + noise_eu
    )
    
    df_eu = pd.DataFrame({
        "Region": "EU",
        "TV_Budget": tv_eu,
        "Radio_Budget": radio_eu,
        "SocialMedia_Budget": social_eu,
        "Is_Holiday": holiday_eu,
        "Sales": sales_eu
    })

    # ==========================================
    # 3. 合并、洗牌并导出
    # ==========================================
    # 合并两个市场
    df_final = pd.concat([df_na, df_eu], ignore_index=True)
    
    # 随机打乱顺序，防止学生发现上半部分全是NA，下半部分全是EU
    df_final = df_final.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    
    # 保留两位小数，让数据看起来更像真实的业务系统导出报表
    df_final = df_final.round(2)
    
    # 创建目录并保存
    file_path.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(file_path, index=False)
    print(f"✅ 成功生成真实业务数据集：{file_path} (共 {len(df_final)} 行)")

if __name__ == "__main__":
    # 在当前目录生成 data/q3_marketing.csv
    output_path = Path("data/q3_marketing.csv")
    generate_marketing_data(output_path)