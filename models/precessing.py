import pandas as pd

from sklearn.preprocessing import LabelEncoder

raw_train = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\train.csv")
raw_test = pd.read_csv("C:\\Users\\solba\\dacon-project\\data\\raw\\test.csv")

def train_preprocess_data(data):
    personal = data[['ID', 'school1', 'job', 're_registration', 'certificate_acquisition']]
    def count_certificates(text):
        if pd.isna(text) or text == '없음':
            return 0
        return len([cert for cert in text.split(',') if cert.strip()])
    personal['자격증보유개수'] = personal['certificate_acquisition'].apply(count_certificates)
    personal_df = personal[['ID', 'school1', 'job', 're_registration', '자격증보유개수']]

    le = LabelEncoder()

    personal_df['job'] = le.fit_transform(personal_df['job'])
    personal_df['re_registration'] = le.fit_transform(personal_df['re_registration'])

    personal_df = personal_df.rename(columns={
        'school1': '학교',
        'job': '직업',
        're_registration': '재등록'
    })

    major = data[['ID', 'major type', 'major1_1', 'major1_2', 'major_field']]
    high_tech_list = ['IT (컴퓨터 공학 포함)', '공학 (컴퓨터 공학 제외)']
    data_friendly_list = ['경영학', '경제통상학', '자연과학', '자연고학']
    def classify_majors(text):
        if pd.isna(text):
            return pd.Series([0, 0, 0])
        
        majors = [m.strip() for m in text.split(',')]
        
        high_tech = 0
        data_friendly = 0
        others = 0
        
        for m in majors:
            if m in high_tech_list:
                high_tech = 1
            elif m in data_friendly_list:
                data_friendly = 1
            else:
                if m and m != 'nan':
                    others = 1
                    
        return pd.Series([high_tech, data_friendly, others])

    major[['High Tech', 'Data Friendly', 'Others']] = major['major_field'].apply(classify_majors)

    major_df = major[['ID', 'High Tech', 'Data Friendly', 'Others']]

    classs = data[['ID', 'class1', 'class2', 'class3', 'class4']]
    classs_count = classs.set_index('ID').count(axis=1).reset_index()
    classs_count.columns = ['ID', 'classs_cnt']
    previous = data[['ID', 'previous_class_3', 'previous_class_4', 'previous_class_5',
                     'previous_class_6', 'previous_class_7', 'previous_class_8']]
    def count_valid_classes(row):
        return sum((pd.notna(val)) and (val != '해당없음') for val in row)
    prev_rows = previous.set_index('ID')
    prev_count = prev_rows.apply(count_valid_classes, axis=1).reset_index()
    prev_count.columns = ['ID', 'prev_cnt']
    total_df = pd.merge(classs_count, prev_count, on='ID', how='outer').fillna(0)
    total_df['수강개수'] = total_df['classs_cnt'] + total_df['prev_cnt']
    class_df = total_df[['ID', '수강개수']]

    time_df = data[['ID', 'time_input']]
    answer = data[['ID', 'completed']]
    dfs = [personal_df, class_df, time_df, answer]

    integrated_df = personal_df.copy()

    for df in [class_df, time_df, answer]:
        integrated_df = pd.merge(integrated_df, df, on='ID', how='left')

    integrated_df = integrated_df.fillna(0)

    integrated_df = integrated_df.rename(columns={
        'time_input': '투자시간',
        'completed': '수료여부'
    })

    cols_to_convert = integrated_df.columns.drop('ID')

    integrated_df[cols_to_convert] = integrated_df[cols_to_convert].fillna(0).astype(int)

    return integrated_df

def test_preprocess_data(data):
    personal = data[['ID', 'school1', 'job', 're_registration', 'certificate_acquisition']]
    def count_certificates(text):
        if pd.isna(text) or text == '없음':
            return 0
        return len([cert for cert in text.split(',') if cert.strip()])
    personal['자격증보유개수'] = personal['certificate_acquisition'].apply(count_certificates)
    personal_df = personal[['ID', 'school1', 'job', 're_registration', '자격증보유개수']]

    le = LabelEncoder()

    personal_df['job'] = le.fit_transform(personal_df['job'])
    personal_df['re_registration'] = le.fit_transform(personal_df['re_registration'])

    personal_df = personal_df.rename(columns={
        'school1': '학교',
        'job': '직업',
        're_registration': '재등록'
    })

    major = data[['ID', 'major type', 'major1_1', 'major1_2', 'major_field']]
    high_tech_list = ['IT (컴퓨터 공학 포함)', '공학 (컴퓨터 공학 제외)']
    data_friendly_list = ['경영학', '경제통상학', '자연과학', '자연고학']
    def classify_majors(text):
        if pd.isna(text):
            return pd.Series([0, 0, 0])
        
        majors = [m.strip() for m in text.split(',')]
        
        high_tech = 0
        data_friendly = 0
        others = 0
        
        for m in majors:
            if m in high_tech_list:
                high_tech = 1
            elif m in data_friendly_list:
                data_friendly = 1
            else:
                if m and m != 'nan':
                    others = 1
                    
        return pd.Series([high_tech, data_friendly, others])

    major[['High Tech', 'Data Friendly', 'Others']] = major['major_field'].apply(classify_majors)

    major_df = major[['ID', 'High Tech', 'Data Friendly', 'Others']]

    classs = data[['ID', 'class1', 'class2', 'class3', 'class4']]
    classs_count = classs.set_index('ID').count(axis=1).reset_index()
    classs_count.columns = ['ID', 'classs_cnt']
    previous = data[['ID', 'previous_class_3', 'previous_class_4', 'previous_class_5',
                     'previous_class_6', 'previous_class_7', 'previous_class_8']]
    def count_valid_classes(row):
        return sum((pd.notna(val)) and (val != '해당없음') for val in row)
    prev_rows = previous.set_index('ID')
    prev_count = prev_rows.apply(count_valid_classes, axis=1).reset_index()
    prev_count.columns = ['ID', 'prev_cnt']
    total_df = pd.merge(classs_count, prev_count, on='ID', how='outer').fillna(0)
    total_df['수강개수'] = total_df['classs_cnt'] + total_df['prev_cnt']
    class_df = total_df[['ID', '수강개수']]

    time_df = data[['ID', 'time_input']]
    dfs = [personal_df, class_df, time_df]

    integrated_df = personal_df.copy()

    for df in [class_df, time_df]:
        integrated_df = pd.merge(integrated_df, df, on='ID', how='left')

    integrated_df = integrated_df.fillna(0)

    integrated_df = integrated_df.rename(columns={
        'time_input': '투자시간'
    })

    cols_to_convert = integrated_df.columns.drop('ID')

    integrated_df[cols_to_convert] = integrated_df[cols_to_convert].fillna(0).astype(int)

    return integrated_df
