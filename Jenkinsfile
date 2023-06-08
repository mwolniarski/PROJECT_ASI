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
                    docker.image('project_asi').run('-d --network=project_asi -v wandb_logs:/home/kedro_docker/wandb')
                }
            }
        }
    }
}