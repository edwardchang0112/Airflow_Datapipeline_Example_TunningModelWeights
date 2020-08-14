from datetime import datetime, timedelta
from airflow import DAG
from main_task import *
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.python_operator import BranchPythonOperator, PythonOperator

default_args = {
    'owner': 'someone',
    'depends_on_past': False,
    'start_date': datetime(2020, 2, 24),
    'email': ['edward@example.com'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
    # 'end_date': datetime(2020, 2, 29),
    # 'execution_timeout': timedelta(seconds=300),
    # 'on_failure_callback': some_function,
    # 'on_success_callback': some_other_function,
    # 'on_retry_callback': another_function,
}

dag = DAG(
    dag_id='mytask_dag',
    description='mytask_dag',
    default_args=default_args,
    schedule_interval='*/1 * * * *'
)

task1 = PythonOperator(
    task_id='Load_data',
    python_callable=read_excel,
    op_args=['YOURPATH/airflow/dags/user_data.xlsx'], # must use absolute path
    dag=dag
)

def loadDatatoLoop(**context):
    df = context['task_instance'].xcom_pull(task_ids='Load_data')
    good_learning_rate_default, good_n_estimators_default, good_max_depth_default, good_gamma_default = mainTask(df)
    return good_learning_rate_default, good_n_estimators_default, good_max_depth_default, good_gamma_default

task2 = PythonOperator(
    task_id='Loop',
    provide_context=True, # to get values from other function
    python_callable=loadDatatoLoop,
    dag=dag
)

def paramstoStore(**context):
    good_learning_rate_default, good_n_estimators_default, good_max_depth_default, good_gamma_default = context['task_instance'].xcom_pull(task_ids='Loop')
    status = store_excel(good_learning_rate_default, good_n_estimators_default, good_max_depth_default, good_gamma_default)
    return status

task3 = PythonOperator(
    task_id='Store_weights',
    provide_context=True,
    python_callable=paramstoStore,
    dag=dag
)

task1 >> task2 >> task3