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
                    docker.image('project_asi').run('-d --link 35ca663b3a1ecd292b6d226a637218f5cca556d1b37fece2cf2fc2b42d8c49cb:mlflow')
                }
            }
        }
    }
}