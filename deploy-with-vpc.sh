#!/bin/bash

# Load environment variables from .env file
if [ -f .env ]; then
    export $(cat .env | grep -v '#' | sed 's/\r$//' | awk '/=/ {print $1}' )
else
    echo ".env file not found"
    exit 1
fi

# Set your AWS region
AWS_REGION=${AWS_REGION:-"us-east-1"}

# Set your stack name
STACK_NAME=${STACK_NAME:-"RDSPGVectorStack"}

# Set the path to your CloudFormation template
TEMPLATE_PATH=${TEMPLATE_PATH:-"aws-infrastructure/rds-pgvector-setup-with-vpc.yaml"}

# Use environment variables for sensitive data, with defaults for non-sensitive data
DB_INSTANCE_CLASS=${DB_INSTANCE_CLASS:-"db.t4g.micro"}
ALLOCATED_STORAGE=${ALLOCATED_STORAGE:-"20"}
DB_USERNAME=${DB_USERNAME:?DB_USERNAME is required}
DB_PASSWORD=${DB_PASSWORD:?DB_PASSWORD is required}
DB_NAME=${DB_NAME:-"n8n-rag"}
VPC_CIDR=${VPC_CIDR:-"10.0.0.0/16"}
PUBLIC_SUBNET_CIDR1=${PUBLIC_SUBNET_CIDR1:-"10.0.1.0/24"}
PUBLIC_SUBNET_CIDR2=${PUBLIC_SUBNET_CIDR2:-"10.0.2.0/24"}
PRIVATE_SUBNET_CIDR1=${PRIVATE_SUBNET_CIDR1:-"10.0.3.0/24"}
PRIVATE_SUBNET_CIDR2=${PRIVATE_SUBNET_CIDR2:-"10.0.4.0/24"}
MULTI_AZ=${MULTI_AZ:-"false"}
AWS_PROFILE=${AWS_PROFILE:-"default"}

# Create the stack
aws cloudformation create-stack \
  --stack-name $STACK_NAME \
  --template-body file://$TEMPLATE_PATH \
  --region $AWS_REGION \
  --parameters \
    ParameterKey=DBInstanceClass,ParameterValue=$DB_INSTANCE_CLASS \
    ParameterKey=AllocatedStorage,ParameterValue=$ALLOCATED_STORAGE \
    ParameterKey=DBUsername,ParameterValue=$DB_USERNAME \
    ParameterKey=DBPassword,ParameterValue=$DB_PASSWORD \
    ParameterKey=DBName,ParameterValue=$DB_NAME \
    ParameterKey=VPCCIDR,ParameterValue=$VPC_CIDR \
    ParameterKey=PublicSubnetCIDR1,ParameterValue=$PUBLIC_SUBNET_CIDR1 \
    ParameterKey=PublicSubnetCIDR2,ParameterValue=$PUBLIC_SUBNET_CIDR2 \
    ParameterKey=PrivateSubnetCIDR1,ParameterValue=$PRIVATE_SUBNET_CIDR1 \
    ParameterKey=PrivateSubnetCIDR2,ParameterValue=$PRIVATE_SUBNET_CIDR2 \
    ParameterKey=MultiAZ,ParameterValue=$MULTI_AZ \
  --capabilities CAPABILITY_IAM \
  --profile $AWS_PROFILE

echo "Stack creation initiated. Check the AWS CloudFormation console for status updates."
