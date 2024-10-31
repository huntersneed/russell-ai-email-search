# AI Email Search and Analysis Tool

This repository contains a powerful AI-driven email search and analysis tool that uses Retrieval-Augmented Generation (RAG) to provide intelligent responses to queries about your email data. It includes AWS infrastructure setup for deploying a PostgreSQL database with the pgvector extension, as well as utility scripts for database management.

**Note: Currently, this tool only supports Office 365 email accounts and requires an Azure App Registration.**


## Features

- Email parsing and embedding (Office 365 only)
- Natural language querying of email content
- Customizable AI model selection (OpenAI and Anthropic)
- Metadata filtering for refined searches
- Interactive chat interface with Streamlit
- AWS RDS PostgreSQL instance with pgvector extension
- Utility scripts for database management and troubleshooting
- Automated deployment script for AWS infrastructure

## Components

### 1. Email Downloader

Located in `email_scraping/download_emails.py`, this script:

- Authenticates with Microsoft Graph API using device flow
- Downloads emails from Office 365 inbox and sent folders
- Saves emails in .eml format to local directories
- Implements retry logic and rate limiting
- Handles batch processing for efficient downloads

### 2. Email Parser and Embedder

Located in `email_scraping/office365_email_parser.py`, this script:

- Parses downloaded .eml files using UnstructuredEmailLoader
- Extracts comprehensive email metadata (sender, recipients, dates, etc.)
- Chunks email content with configurable sizes and overlap
- Calculates token usage and estimated costs
- Embeds email content using OpenAI's embedding models
- Stores embeddings and metadata in PostgreSQL with pgvector
- Supports multiple embedding models (text-embedding-3-small/large, ada_v2)
- Implements connection pooling and error handling
- Provides verification of stored embeddings

### 3. Frontend Chat Interface

Located in `frontend/app.py`, this Streamlit application provides:

- An interactive chat interface for querying email data
- Model selection between OpenAI and Anthropic
- Metadata filters for sender, subject, and date range
- Display of relevant email sources for each query

### 4. AWS Infrastructure

- `aws-infrastructure/rds-pgvector-setup-with-vpc.yaml`: CloudFormation template for setting up RDS with a new VPC
- `aws-infrastructure/rds-pgvector-setup-without-vpc.yaml`: CloudFormation template for setting up RDS in an existing VPC

These templates set up:
- An RDS PostgreSQL instance with pgvector extension
- Necessary security groups and subnet groups
- Customizable database parameters

### 5. Deployment Script

Located at `deploy-with-vpc.sh`, this script automates the deployment process:

- Checks for existing VPC and subnets
- Creates or updates the CloudFormation stack
- Sets up the pgvector extension in the RDS instance

### 6. Utility Scripts

Located in the `utils` directory:

- `clear_vector_table.py`: Allows clearing specific tables in the database
- `check_db_connection.py`: Verifies the database connection and checks for pgvector extension
- `view_vector_db.py`: Displays the contents of specified tables in the vector database

## Setup and Usage

### 1. AWS Infrastructure Deployment

1. Ensure you have the AWS CLI installed and configured with the necessary credentials.
2. Create a `.env` file in the root directory with the required environment variables (see Environment Variables section).
3. Run the deployment script:
   ```
   ./deploy-with-vpc.sh
   ```
   This script will create or update the CloudFormation stack and set up the RDS instance with pgvector.

### 2. Azure App Registration

1. Go to the [Azure Portal](https://portal.azure.com/).
2. Navigate to "Azure Active Directory" > "App registrations".
3. Click on "New registration".
4. Name your application and select "Accounts in this organizational directory only".
5. Set the redirect URI to "http://localhost" (type: Web).
6. Click "Register".
7. Once created, note down the "Application (client) ID" and "Directory (tenant) ID".
8. Go to "Certificates & secrets", create a new client secret, and note it down.
9. Go to "API permissions" and add the following permissions:
   - Microsoft Graph API
   - Delegated permissions
   - Mail.Read
   - User.Read
10. Click "Grant admin consent" for your organization.

### 3. Application Setup

1. Clone the repository
2. Install required dependencies (requirements.txt file needed)
3. Set up your `.env` file with the required variables, including Azure App Registration details and AWS infrastructure details
4. Verify database connection:
   ```
   python utils/check_db_connection.py
   ```
5. Download your emails:
   ```
   python email_scraping/download_emails.py
   ```
   This will create `Emails/Inbox` and `Emails/Sent` directories with your .eml files.
6. Process and embed the emails:
   ```
   python email_scraping/office365_email_parser.py --parse --embed
   ```
   You can customize the embedding process with additional arguments:
   - `--chunk-size`: Size of text chunks (default: 1000)
   - `--chunk-overlap`: Overlap between chunks (default: 200)
   - `--model`: Embedding model to use (default: text-embedding-3-small)
7. Launch the Streamlit app:
   ```
   streamlit run frontend/app.py
   ```

### Using Utility Scripts

- To check database connection:
  ```
  python utils/check_db_connection.py
  ```
- To view vector database contents:
  ```
  python utils/view_vector_db.py
  ```
- To clear specific tables:
  ```
  python utils/clear_vector_table.py
  ```

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```
DB_USERNAME=your_db_username
DB_PASSWORD=your_db_password
RDS_ENDPOINT=your_rds_endpoint_from_cloudformation_output
RDS_PORT=your_rds_port_from_cloudformation_output
DB_NAME=your_db_name
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=your_anthropic_api_key
AWS_REGION=your_aws_region
STACK_NAME=your_cloudformation_stack_name
DB_INSTANCE_CLASS=db.t4g.micro
ALLOCATED_STORAGE=20
VPC_CIDR=10.0.0.0/16
PUBLIC_SUBNET_CIDR1=10.0.1.0/24
PUBLIC_SUBNET_CIDR2=10.0.2.0/24
PRIVATE_SUBNET_CIDR1=10.0.3.0/24
PRIVATE_SUBNET_CIDR2=10.0.4.0/24
MULTI_AZ=false
AWS_PROFILE=default

# Azure App Registration details
AZURE_CLIENT_ID=your_azure_client_id
AZURE_CLIENT_SECRET=your_azure_client_secret
AZURE_TENANT_ID=your_azure_tenant_id
```

## AWS CloudFormation Template Parameters

The CloudFormation templates include several customizable parameters. Refer to the template files for details on available parameters and their descriptions.

## Note

- This tool currently only supports Office 365 email accounts.
- Ensure that you have completed the Azure App Registration process and have the necessary credentials.
- Ensure that you have the necessary API keys for OpenAI and Anthropic services.
- The AWS CloudFormation template creates resources that may incur costs. Be sure to review the AWS pricing for the services used.
- The template sets up the RDS instance to be publicly accessible for testing purposes. For production use, consider modifying the security group settings and network architecture for enhanced security.
- Use the utility scripts with caution, especially `clear_vector_table.py`, as it can delete data from your database.

## Future Improvements

- Add support for other email providers (Gmail, IMAP, etc.)
- Implement OAuth flow for easier Office 365 authentication
- Enhance security measures for production deployment

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## Estimated Infrastructure Costs

This section provides an estimate of the monthly recurring infrastructure costs for running this project. Please note that these are approximate figures and can vary based on usage patterns and potential pricing changes by service providers.

### Monthly Recurring Costs

1. **AWS RDS PostgreSQL with pgvector**:
   - DB Instance: db.t4g.micro
   - Storage: 20 GB
   - Estimated Cost: $15 - $25/month

2. **OpenAI API** (usage-based):
   - Embedding: $0.0001 per 1K tokens
   - GPT-3.5-turbo: $0.002 per 1K tokens
   - Estimated Cost: $12 - $20/month
     (Assuming 100,000 emails with an average of 1,000 tokens each for embedding, and 1,000 queries per month)

3. **Anthropic API** (estimated):
   - Estimated Cost: $5 - $15/month

4. **AWS Data Transfer**:
   - Estimated Cost: $1 - $5/month

5. **Azure Active Directory** (for App Registration):
   - Free tier (sufficient for this project)

**Total Estimated Monthly Infrastructure Costs: $33 - $65/month**

### Additional Considerations

- Costs will scale with increased email volume and query frequency.
- AWS Free Tier benefits may apply for the first 12 months if eligible.
- Consider AWS Reserved Instances for long-term cost optimization.
- Data storage costs will grow as more emails are added over time.

It is recommended to set up AWS billing alerts and closely monitor API usage, as actual costs can vary significantly based on real-world usage patterns.

## License

[MIT License](LICENSE)
