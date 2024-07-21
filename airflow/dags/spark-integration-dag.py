from airflow import DAG
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.bash_operator import BashOperator
from datetime import datetime

default_args = {
    'owner': 'airflow',
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

dag = DAG('spark_integration_dag', default_args=default_args, schedule_interval='@daily')

start = DummyOperator(task_id='start', dag=dag)

spark_job = BashOperator(
    task_id='run_spark_job',
    bash_command='spark-submit /spark/jobs/spark-jobs.py',
    dag=dag
)

end = DummyOperator(task_id='end', dag=dag)

start >> spark_job >> end
