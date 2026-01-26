import pandas as pd
from sklearn.preprocessing import LabelEncoder

def column_drop(data):
    columns_to_drop = ['generation', 'contest_award', 'completed_semester', 'major_data',  
                    'incumbents_lecture_scale_reason', 'interested_company', 'contest_participation', 'idea_contest']
    data = data.drop(columns = columns_to_drop)
    return data

def personal(data):
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

    data[['High Tech', 'Data Friendly', 'Others']] = data['major_field'].apply(classify_majors)

    personal_df = data[['ID', 'school1', 'job', 'nationality', 'High Tech', 'Data Friendly', 'Others']]

    return personal_df

def previous(data):
    def count_valid(row):
        count = 0
        for col in row.index:
            if col == 'certificate_acquisition':
                val = row[col]
                if pd.notna(val) and val not in ['해당없음', '없음']:
                    # 여러 자격증이 콤마로 구분되어 있을 수 있음
                    count += len([x for x in str(val).split(',') if x.strip() and x.strip() not in ['해당없음', '없음']])
            elif col != 'ID':
                val = row[col]
                if pd.notna(val) and val not in ['해당없음', '없음']:
                    count += 1
        return count
    
    previous = data[['ID', 'class1', 'class2', 'class3', 'class4', 'previous_class_3', 'previous_class_4', 'previous_class_5',
                    'previous_class_6', 'previous_class_7', 'previous_class_8', 'certificate_acquisition']]
    previous_df = previous.apply(count_valid, axis=1)
    previous_df = pd.DataFrame({'ID': previous['ID'], 'count': previous_df})
    return previous_df

def want_in(data):
    # 원하는 것 개수 세기 함수 (expected_domain은 맨 앞 알파벳만 세기)
    def count_want(row):
        count = 0
        for col in ['desired_job', 'desired_job_except_data', 'desired_certificate']:
            val = row[col]
            if pd.isna(val):
                continue
            items = [x.strip() for x in str(val).split(',') if x.strip() and x.strip() not in ['없음', '아직 없음']]
            count += len(items)
        # expected_domain은 맨 앞 알파벳만 세기
        val = row['expected_domain']
        if pd.notna(val):
            # 예: 'J. 정보통신업, O. 공공 행정, 국방 및 사회보장 행정' → ['J', 'O']
            items = [x.strip() for x in str(val).split(',') if x.strip()]
            domain_initials = [x.split('.')[0].strip() for x in items if '.' in x]
            count += len(domain_initials)
        return count

    data['want_count'] = data.apply(count_want, axis=1)
    want_int = data[['ID', 'want_count']]
    return want_int

def time_input(data):
    input_df = data[['ID', 'time_input']]
    return input_df

def want_st(data):
    want_str = data[['ID', 'hope_for_group', 'desired_career_path']]
    return want_str

def teacher(data):
    teacher_df = data[['ID', 'incumbents_level', 'incumbents_lecture', 'incumbents_company_level', 'incumbents_lecture_type', 'incumbents_lecture_scale']]
    return teacher_df

def merge_df(data):
    data = column_drop(data)
    personal_df = personal(data)
    want_st_df = want_st(data)
    teacher_df = teacher(data)
    previous_df = previous(data)
    time_df = time_input(data)
    want_in_df = want_in(data)

    merged_df = personal_df.merge(want_st_df, on='ID', how='left') \
        .merge(teacher_df, on='ID', how='left') \
        .merge(previous_df, on='ID', how='left') \
        .merge(time_df, on='ID', how='left') \
        .merge(want_in_df, on='ID', how='left')

    # str로 변환할 컬럼: train_personal_df, wantst_train_df, train_teacher_df의 컬럼들

    # int로 변환할 컬럼: previous_df, time_df, want_in_df의 컬럼들
    int_cols = list(previous_df.columns) + list(time_df.columns) + list(want_in_df.columns)
    int_cols = [col for col in int_cols if col != 'ID']

    # str로 변환할 컬럼: personal_df, want_st_df, teacher_df의 컬럼들
    str_cols = list(personal_df.columns) + list(want_st_df.columns) + list(teacher_df.columns)
    str_cols = [col for col in str_cols if col != 'ID']
    for col in str_cols:
        if col in merged_df.columns:
            merged_df[col] = merged_df[col].astype(str)

    for col in int_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
            if pd.api.types.is_float_dtype(merged_df[col]):
                merged_df[col] = merged_df[col].round().astype('Int64')

    le = LabelEncoder()
    for col in merged_df.columns[2:4]:
        merged_df[col] = le.fit_transform(merged_df[col].astype(str))
    for col in merged_df.columns[6:14]:
        merged_df[col] = le.fit_transform(merged_df[col].astype(str))

    return merged_df

def completed(data, merged_df):
    Y = data[['ID', 'completed']]
    merged_df = merged_df.merge(Y, on = 'ID', how = 'left')
    return merged_df

