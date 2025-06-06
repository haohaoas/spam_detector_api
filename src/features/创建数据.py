import pandas as pd

df_main = pd.read_csv('../../data/spam_enhanced_labeled.csv', encoding='utf-8-sig')  # 这是你合并过qqmail后的主数据
df_official = pd.read_csv('../../data/official_verification_ham.csv', encoding='utf-8-sig')
df_official['lang'] = 'chinese'

df_all = pd.concat([df_main, df_official], ignore_index=True)
df_all.to_csv('spam_enhanced_labeled_merged_v2.csv', index=False, encoding='utf-8-sig')
print(f"已合并官方业务ham后共 {len(df_all)} 条数据，存为 spam_enhanced_labeled_merged_v2.csv")