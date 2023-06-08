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
                    def container = docker.image('project_asi').run()
                    def containerIP = sh(script: "docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' ${container.id}", returnStdout: true).trim()
                    sh "docker exec vigilant_lichterman ping ${containerIP}"
                }
            }
        }
    }
}