from kfp import dsl
from kfp.v2 import compiler
from kfp.v2.google.client import AIPlatformClient
from kfp.v2.dsl import component
from google.cloud import aiplatform


@component(packages_to_install=["torch", "torchvision"])
def train_resnet18_on_cifar10():
    import torch
    import torchvision
    import torchvision.transforms as transforms

    # Load the CIFAR10 training and test datasets
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=4, shuffle=True, num_workers=2
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=4, shuffle=False, num_workers=2
    )

    # Define the ResNet18 model
    net = torchvision.models.resnet18()

    # Define a Loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Train the network
    for epoch in range(2):  # Loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Save the trained model
    torch.save(net.state_dict(), "./resnet18_cifar10.pth")


# Define the pipeline
@dsl.pipeline(
    name="ResNet18-on-CIFAR10",
    description="Train a ResNet18 model on the CIFAR-10 dataset using PyTorch",
)
def resnet_pipeline():
    train_resnet18_on_cifar10_op = train_resnet18_on_cifar10()


# Compile the pipeline
compiler.Compiler().compile(
    pipeline_func=resnet_pipeline, package_path="resnet_pipeline.json"
)

# Instantiate the Vertex AI client
client = AIPlatformClient(
    project_id="your-project-id",
    region="your-region",
)

# Submit the pipeline job
response = client.create_run_from_job_spec(
    job_spec_path="resnet_pipeline.json",
    parameter_values={},
)
