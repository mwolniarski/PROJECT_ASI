pipeline {
    agent any
    stages {
        stage('Checkout') {
            steps {
                checkout scm
            }
        }
        stage('Build Docker Image') {
            steps {
                script {
                    docker.build('project_asi')
                }
            }
        }
        stage('Run Docker Container') {
            steps {
                script {
                    docker.image('project_asi').run('--name=kedro_docker --network=project_asi -v wandb_logs:/home/kedro_docker/wandb -v mlflow_artifact:/home/kedro_docker/mlruns')
                }
            }
        }
        stage('Remove container') {
            steps {
                script {
                    dockerContainerRemove(container: 'project_asi')
                }
            }
        }
    }
}