# This will create both resource group and workspace
# az ml workspace create \
#   --name "ai-customer-service" \
#   --resource-group "AI-Group" \
#   --location "eastus" \
#   --create-resource-group

from azure.ai.ml import MLClient
from azure.core.exceptions import HttpResponseError, ResourceNotFoundError
from azure.ai.ml.entities import ManagedOnlineEndpoint, ManagedOnlineDeployment, Environment, CodeConfiguration, Model
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import uuid
import time
import os, json

load_dotenv()



SUBSCRIPTION = os.getenv("YOUR_SUBSCRIPTION_ID")
RESOURCE_GROUP = "aicso-dev-rg"
WORKSPACE = "aicso-dev-ml-ws"

ENDPOINT_FILE = "deployed_endpoint.json"

def save_endpoint_name(endpoint_name):
    """Save endpoint name to file for reuse"""
    with open(ENDPOINT_FILE, 'w') as f:
        json.dump({"endpoint_name": endpoint_name}, f)
    print(f"Saved endpoint name to {ENDPOINT_FILE}")

def load_endpoint_name():
    """Load endpoint name from file if exists"""
    if os.path.exists(ENDPOINT_FILE):
        with open(ENDPOINT_FILE, 'r') as f:
            data = json.load(f)
            return data.get("endpoint_name", "")
    return None

def endpoint_exists(ml_client, endpoint_name):
    """Check if endpoint already exists"""
    try:
        ml_client.online_endpoints.get(name=endpoint_name)
        print('Testing 1,2,3')
        return True
    except ResourceNotFoundError:
        return False

try:
    # --- Initialize ML Client ---
    ml_client = MLClient(DefaultAzureCredential(), SUBSCRIPTION, RESOURCE_GROUP, WORKSPACE)
    print("Connected to Azure ML workspace")

    existing_endpoint = load_endpoint_name()
    if existing_endpoint and endpoint_exists(ml_client, existing_endpoint):
        use_existing = input(f"Found existing endpoint '{existing_endpoint}'. Use it? (y/n): ")
        if use_existing.lower() == 'y':
            endpoint_name = existing_endpoint
            print(f"Using existing endpoint: {endpoint_name}")
            
            # Get endpoint details
            endpoint = ml_client.online_endpoints.get(endpoint_name)
            keys = ml_client.online_endpoints.get_keys(name=endpoint_name)
            
            print(f"\nENDPOINT READY!")
            print(f"Endpoint: {endpoint_name}")
            print(f"Scoring URI: {endpoint.scoring_uri}")
            print(f"Primary Key: {keys.primary_key}")
            
            print(f"\nTo test:")
            print(f'curl -X POST {endpoint.scoring_uri} \\')
            print(f'  -H "Authorization: Bearer {keys.primary_key}" \\')
            print(f'  -H "Content-Type: application/json" \\')
            print(f'  -d \'{{"text": "I want to check my balance"}}\'')
            exit(0)
        else:
            # User wants new endpoint
            existing_endpoint = None

    # --- Use existing model and creaste environment ---
    # model = ml_client.models.get(name="bank-intent-model", version="1")
    
    # --- Register model if not exists ---
    model = Model(
        path="bank_model.pkl",
        name="bank-intent-model",
        version="1",
        type="custom_model"
    )

    ml_client.models.create_or_update(model)
    print("Model registered successfully!")

    env = Environment(
        conda_file="environment.yml",
        image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest",
    )
    
    print(f"Using model: {model.name} v{model.version}")
    print(f"Using environment: {env.name} v{env.version}")

    # --- Create endpoint ---
    if existing_endpoint:
        print(existing_endpoint)
        endpoint_name = existing_endpoint
        print(f"Using existing endpoint: {endpoint_name}")
    else:
        endpoint_name = f"bank-live-{uuid.uuid4().hex[:6]}"
        print(f"Creating new endpoint: {endpoint_name}")
        
        endpoint = ManagedOnlineEndpoint(
            name=endpoint_name, 
            auth_mode="key",
            description="Bank intent classification endpoint"
        )

        # --- Create the endpoint ---
        print("Creating endpoint...")
        endpoint_poller = ml_client.begin_create_or_update(endpoint)
        endpoint_result = endpoint_poller.result()
        print(f"Endpoint created: {endpoint_name}")
        
        # SAVE THE ENDPOINT NAME
        save_endpoint_name(endpoint_name)

    # Wait for endpoint to be ready
    print("Waiting for endpoint to be ready...")
 
    time.sleep(20)

    try:
        existing_deployment = ml_client.online_deployments.get(
            name="blue", 
            endpoint_name=endpoint_name
        )
        print("Deployment 'blue' already exists")
        deployment_exists = True
    except ResourceNotFoundError:
        deployment_exists = False

    if not deployment_exists:

        instance_types_to_try = [
        "Standard_DS2_v2",
        "Standard_DS3_v2", 
        "Standard_D2s_v3"
    ]

        successful_deployment = False

        for instance_type in instance_types_to_try:
            try:
                print(f"Trying instance type: {instance_type}")
                
                deployment = ManagedOnlineDeployment(
                    name="blue",
                    endpoint_name=endpoint_name,
                    model=model.id,
                    environment=env,
                    code_configuration=CodeConfiguration(
                    code=".", scoring_script="score.py"
                    ),
                    instance_type=instance_type,
                    instance_count=1,
                )

                deployment_poller = ml_client.online_deployments.begin_create_or_update(deployment, local=False, timeout=3600)
                deployment_result = deployment_poller.result()
                print(f"Deployment successful with {instance_type}!")
                successful_deployment = True
                break
                
            except HttpResponseError as e:
                if "quota" in str(e).lower() or "outofquota" in str(e).lower():
                    print(f"Quota exceeded for {instance_type}, trying next...")
                    continue
                else:
                    print(e)
                    continue

        if not successful_deployment:
            print("All deployment attempts failed")
            exit(1)
    else:
        print("Using Existing deployment")
        


    # --- Set traffic ---
    print("Setting traffic to 100% to 'blue' deployment...")
    endpoint.traffic = {"blue": 100}
    ml_client.online_endpoints.begin_create_or_update(
        endpoint,    
    ).result()

    endpoint = ml_client.online_endpoints.get(endpoint_name)
    keys = ml_client.online_endpoints.get_keys(name=endpoint_name)

    print(f"\nDEPLOYMENT SUCCESSFUL!")
    print(f"Endpoint: {endpoint_name}")
    print(f"Scoring URI: {endpoint.scoring_uri}")
    print(f"Primary Key: {keys.primary_key}")
    
    print(f"\nTest command:")
    print(f'curl -X POST {endpoint.scoring_uri} \\')
    print(f'  -H "Authorization: Bearer {keys.primary_key}" \\')
    print(f'  -H "Content-Type: application/json" \\')
    print(f'  -d \'{{"text": "I want to check my balance"}}\'')

except HttpResponseError as e:
    print(f"Azure Error: {e.message}")
    if "SubscriptionNotRegistered" in str(e):
        print("\nSOLUTION: Resource providers not registered yet.")
        print("Run these commands and wait 10-15 minutes:")
        print("az provider register --namespace Microsoft.Network")
        print("az provider register --namespace Microsoft.Compute") 
        print("az provider register --namespace Microsoft.ContainerInstance")
        print("az provider register --namespace Microsoft.Cdn")
        print("az provider register --namespace Microsoft.Policies")
        print("\nThen run this script again.")
    elif "QuotaExceeded" in str(e):
        print("\nTry a smaller instance: Standard_B2s or Standard_B1s")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()
