pipeline {
  agent any

  stages {
    stage('ci') {
      parallel {

        stage('lint') {
          steps {
            sh '''#!/bin/bash
            export PATH=$PATH:$CONDAPATH
            source activate petals_env
            pylint -ry climada_petals | tee pylint.log'''

            recordIssues tools: [pyLint(pattern: 'pylint.log')]
          }
        }

        stage('unit_test') {
          steps {
            sh '''#!/bin/bash
            export PATH=$PATH:$CONDAPATH
            source activate petals_env
            python -m coverage run  tests_runner.py unit
            python -m coverage xml -o coverage.xml
            python -m coverage html -d coverage'''
          }
        }

      }
    }
  }

  post {
    always {
      junit 'tests_xml/*.xml'
      cobertura coberturaReportFile: 'coverage.xml'
    }
  }
}