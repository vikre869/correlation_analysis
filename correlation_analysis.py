from pearson_corr_analysis import calculate_pearson

for i in range(1,11):
    print(f'Analyzing part {i}/10')
    calculate_pearson(10, i)