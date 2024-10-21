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
STACK_NAME=${STACK_NAME:-"RDSPGVectorStack-1"}

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

# Check if VPC exists
echo "Checking if VPC exists..."
EXISTING_VPC_ID=$(aws ec2 describe-vpcs --filters "Name=tag:Name,Values=RDS-VPC" --query "Vpcs[0].VpcId" --output text --region $AWS_REGION --profile $AWS_PROFILE)

if [ -z "$EXISTING_VPC_ID" ] || [ "$EXISTING_VPC_ID" == "None" ]; then
    echo "VPC does not exist. Setting CreateVPC parameter to true."
    CREATE_VPC="true"
    EXISTING_VPC_ID=""
    EXISTING_PUBLIC_SUBNET1_ID=""
    EXISTING_PUBLIC_SUBNET2_ID=""
else
    echo "Found existing VPC: $EXISTING_VPC_ID"
    CREATE_VPC="false"
    
    # Check for existing subnets
    EXISTING_PUBLIC_SUBNET1_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$EXISTING_VPC_ID" "Name=cidr-block,Values=$PUBLIC_SUBNET_CIDR1" --query "Subnets[0].SubnetId" --output text --region $AWS_REGION --profile $AWS_PROFILE)
    EXISTING_PUBLIC_SUBNET2_ID=$(aws ec2 describe-subnets --filters "Name=vpc-id,Values=$EXISTING_VPC_ID" "Name=cidr-block,Values=$PUBLIC_SUBNET_CIDR2" --query "Subnets[0].SubnetId" --output text --region $AWS_REGION --profile $AWS_PROFILE)
    
    if [ -z "$EXISTING_PUBLIC_SUBNET1_ID" ] || [ "$EXISTING_PUBLIC_SUBNET1_ID" == "None" ] || [ -z "$EXISTING_PUBLIC_SUBNET2_ID" ] || [ "$EXISTING_PUBLIC_SUBNET2_ID" == "None" ]; then
        echo "Existing VPC found, but subnets are missing. Setting CreateVPC to true to create new VPC and subnets."
        CREATE_VPC="true"
        EXISTING_VPC_ID=""
        EXISTING_PUBLIC_SUBNET1_ID=""
        EXISTING_PUBLIC_SUBNET2_ID=""
    else
        echo "Found existing subnets: $EXISTING_PUBLIC_SUBNET1_ID, $EXISTING_PUBLIC_SUBNET2_ID"
    fi
fi

# Prepare parameters
PARAMS="\
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
    ParameterKey=CreateVPC,ParameterValue=$CREATE_VPC"

# Add existing VPC and subnet parameters only if they exist
if [ "$CREATE_VPC" = "false" ]; then
    PARAMS="$PARAMS \
    ParameterKey=ExistingVPCId,ParameterValue=$EXISTING_VPC_ID \
    ParameterKey=ExistingPublicSubnet1Id,ParameterValue=$EXISTING_PUBLIC_SUBNET1_ID \
    ParameterKey=ExistingPublicSubnet2Id,ParameterValue=$EXISTING_PUBLIC_SUBNET2_ID"
fi

# Check if the stack already exists
stack_exists=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --region $AWS_REGION --profile $AWS_PROFILE 2>&1)

if echo "$stack_exists" | grep -q 'does not exist'; then
    # Create the stack
    echo "Creating new stack..."
    aws cloudformation create-stack \
      --stack-name $STACK_NAME \
      --template-body file://$TEMPLATE_PATH \
      --region $AWS_REGION \
      --parameters $PARAMS \
      --capabilities CAPABILITY_IAM \
      --profile $AWS_PROFILE

    echo "Waiting for stack creation to complete..."
    aws cloudformation wait stack-create-complete \
      --stack-name $STACK_NAME \
      --region $AWS_REGION \
      --profile $AWS_PROFILE
else
    # Update the stack
    echo "Updating existing stack..."
    aws cloudformation update-stack \
      --stack-name $STACK_NAME \
      --template-body file://$TEMPLATE_PATH \
      --region $AWS_REGION \
      --parameters $PARAMS \
      --capabilities CAPABILITY_IAM \
      --profile $AWS_PROFILE

    echo "Waiting for stack update to complete..."
    aws cloudformation wait stack-update-complete \
      --stack-name $STACK_NAME \
      --region $AWS_REGION \
      --profile $AWS_PROFILE
fi

if [ $? -eq 0 ]; then
  echo "Stack operation completed successfully. Creating pgvector extension..."

  # Get the RDS endpoint
  RDS_ENDPOINT=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='RDSInstanceEndpoint'].OutputValue" \
    --output text \
    --region $AWS_REGION \
    --profile $AWS_PROFILE)

  # Create the pgvector extension
  PGPASSWORD=$DB_PASSWORD psql -h $RDS_ENDPOINT -U $DB_USERNAME -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"

  if [ $? -eq 0 ]; then
    echo "pgvector extension created successfully."
  else
    echo "Failed to create pgvector extension."
  fi
else
  echo "Stack operation failed. Check the AWS CloudFormation console for error details."
fi
