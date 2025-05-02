from MultipleInputAI import Multiple_input_linear_reg_model as MLR
import pandas as pd
import os

df =pd.read_csv(os.path.join(os.getcwd() + '\\taxi_trip_pricing.csv'))

df = df.dropna()
df = df.drop_duplicates()

print(df.info)
print(df.shape)
print(df.head)

Model = MLR(df, ['Trip_Distance_km', 'Trip_Duration_Minutes'], 'Trip_Price', 0.5, 56)
Model.fit()
Model.plot()


ModelAnova = Model.anova()
print(
'Model Anove: \n' +
f'Total sum of sqares : {float(ModelAnova['SST'])}\n' +
f'Regression Sum of Squares: {float(ModelAnova['SSR'])}\n' +
f'Error Sum of Squares: {float(ModelAnova['SSE'])}\n' + 
f'F-statistic: {ModelAnova['F']}\n' +
f'p-value: {ModelAnova['p-value']}\n' + 
'Model is useful: there is relationship between predictors and y\n' if ModelAnova['F'] > ModelAnova['p-value'] else 'Model isn\'t useful: there is **no** relationship between predictors and y\n'
)   

Model_hypothesis_test = Model.hypothesis_test()
print(
'Model Hypothesis test\n' +
f'Model intercept: {Model_hypothesis_test['coefficients'][0]}   Model w0: {Model_hypothesis_test['coefficients'][1]}    Model w2: {Model_hypothesis_test['coefficients'][2]}\n' +
f'Model t-stat for intersept: {Model_hypothesis_test['t_statistics'][0]}    Model t-state for w0: {Model_hypothesis_test['t_statistics'][1]}    Model t-state for w1: {Model_hypothesis_test['t_statistics'][2]}\n' +
f'Model p-value for t-state for intersept: {Model_hypothesis_test['p_values'][0]}   Model p-value for t-state for w1: {Model_hypothesis_test['p_values'][1]}    Model p-value for t-state-for w2: {Model_hypothesis_test['p_values'][2]}\n'+
f'is w0 significant: {'yes' if Model_hypothesis_test['significant'][0] else 'no'}   is w1 significant: {'yes' if Model_hypothesis_test['significant'][1] else 'no'}     is w2 significant: {'yes' if Model_hypothesis_test['significant'][2] else 'no'}\n'+
f'{'Reject' if Model_hypothesis_test['p_values'][0] < Model_hypothesis_test['alpha'] else 'accept'} the hypo. for w0 (w0 is zero)   {'Reject' if Model_hypothesis_test['p_values'][1] < Model_hypothesis_test['alpha'] else 'accept'} the hypo. for w1 (w1 is zero)     {'Reject' if Model_hypothesis_test['p_values'][2] < Model_hypothesis_test['alpha'] else 'accept'} the hypo. for w2 (w2 is zero)'
)

Model_interval_estimation = Model.interval_estimation()
print(
    'Model interval estimation\n'+
    f'w0 lower bound: {Model_interval_estimation[0][0]}     upper bound: {Model_interval_estimation[0][1]}\n' +
    f'w1 lower bound: {Model_interval_estimation[1][0]}     upper bound: {Model_interval_estimation[1][1]}\n' +
    f'w2 lower bound: {Model_interval_estimation[2][0]}     upper bound: {Model_interval_estimation[2][1]}\n' 
)

Model_pridect = Model.pridect(
    pd.DataFrame(
        {
            'Trip_Distance_km': [5, 50, 300, 7, 20],
            'Trip_Duration_Minutes': [10, 20, 30, 40, 50]
        }
    )
)

print('predicted data')
for x in Model_pridect:
    print(
        str(round(float(x),2)) + ' $'
    )


df2 = pd.read_csv(os.path.join(os.getcwd() + '\\Salary_Data.csv'))
df2 = df2.dropna()
df2 = df2.drop_duplicates()

print(df2.info)
print(df2.shape)
print(df2.head)


Model2 = MLR(df2, ['Age', 'Years of Experience'], 'Salary', 0.5, 56)
Model2.fit()
Model2.plot()



Model_pridect = Model2.pridect(
    pd.DataFrame(
        {
            'Age': [30, 50, 60, 40, 20],
            'Years of Experience': [10, 20, 30, 7, 5]
        }
    )
)

print('predicted data')
for x in Model_pridect:
    print(
        str(round(float(x),2)) + ' $'
    )
