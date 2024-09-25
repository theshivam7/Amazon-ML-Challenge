import os
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import requests
from io import BytesIO
from torchvision.models import resnet50, ResNet50_Weights
import torch.nn as nn
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import ast

# Constants
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001
DRIVE_PATH = '/Users/shivamsharma/Desktop/Amazon-ml'
DATASET_FOLDER = os.path.join(DRIVE_PATH, 'dataset')

# Define allowed units
ALLOWED_UNITS = {
    'item_weight': ['gram', 'kilogram', 'ounce', 'pound'],
    'item_length': ['centimetre', 'metre', 'inch', 'foot'],
    # Add other entity types and their allowed units here
}

# Image transformation
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

class ProductImageDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform
        self.entity_types = self.data['entity_name'].unique()
        self.entity_to_idx = {entity: idx for idx, entity in enumerate(self.entity_types)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_url = self.data.iloc[idx]['image_link']
        try:
            response = requests.get(img_url, timeout=10)
            response.raise_for_status()
            img = Image.open(BytesIO(response.content)).convert('RGB')
        except Exception as e:
            print(f"Error loading image from {img_url}: {str(e)}")
            img = Image.new('RGB', (IMAGE_SIZE, IMAGE_SIZE), color='gray')

        if self.transform:
            img = self.transform(img)

        entity_name = self.data.iloc[idx]['entity_name']
        entity_idx = self.entity_to_idx[entity_name]

        if 'entity_value' in self.data.columns:
            value, unit = self.parse_entity_value(self.data.iloc[idx]['entity_value'])
            return img, entity_idx, torch.tensor(value, dtype=torch.float32), unit
        else:
            return img, entity_idx, torch.tensor(0.0), ''

    def parse_entity_value(self, entity_value):
        if pd.isna(entity_value) or entity_value == '':
            return 0.0, ''
        try:
            value_list = ast.literal_eval(entity_value)
            if isinstance(value_list, list):
                if len(value_list) >= 2 and all(isinstance(v, (int, float)) for v in value_list[:2]):
                    value = sum(value_list[:2]) / 2
                else:
                    value = float(value_list[0])
                unit = ' '.join(map(str, value_list[1:] if isinstance(value_list[0], (int, float)) else value_list))
            else:
                raise ValueError
        except (ValueError, SyntaxError):
            parts = str(entity_value).replace('[', '').replace(']', '').split()
            try:
                if len(parts) >= 2 and all(parts[i].replace('.', '').isdigit() for i in range(2)):
                    value = (float(parts[0]) + float(parts[1])) / 2
                else:
                    value = float(parts[0])
                unit = ' '.join(parts[1:])
            except ValueError:
                print(f"Warning: Could not parse entity_value: {entity_value}")
                return 0.0, ''
        return value, unit

class ProductFeatureExtractor(nn.Module):
    def __init__(self, num_entity_types):
        super(ProductFeatureExtractor, self).__init__()
        self.resnet = resnet50(weights=ResNet50_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.fc = nn.Linear(num_ftrs, num_entity_types)

    def forward(self, x, entity_type):
        features = self.resnet(x)
        output = self.fc(features)
        return output.gather(1, entity_type.unsqueeze(1))

def load_data(file_name, possible_paths):
    for path in possible_paths:
        file_path = os.path.join(path, file_name)
        if os.path.exists(file_path):
            print(f"Loading data from: {file_path}")
            return pd.read_csv(file_path)
    raise FileNotFoundError(f"The file {file_name} was not found in any of the specified paths.")

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device):
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for images, entity_idxs, values, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, entity_idxs, values = images.to(device), entity_idxs.to(device), values.to(device)
            optimizer.zero_grad()
            outputs = model(images, entity_idxs)
            loss = criterion(outputs.squeeze(), values)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
        
        train_loss /= len(train_loader.dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, entity_idxs, values, _ in val_loader:
                images, entity_idxs, values = images.to(device), entity_idxs.to(device), values.to(device)
                outputs = model(images, entity_idxs)
                loss = criterion(outputs.squeeze(), values)
                val_loss += loss.item() * images.size(0)
        
        val_loss /= len(val_loader.dataset)
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(DRIVE_PATH, 'best_model.pth'))
    
    print("Training complete.")

def predictor(model, image_link, entity_name, entity_to_idx, device):
    try:
        response = requests.get(image_link, timeout=10)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        img = transform(img).unsqueeze(0).to(device)

        entity_idx = torch.tensor([entity_to_idx[entity_name]], device=device)

        with torch.no_grad():
            output = model(img, entity_idx)

        value = output.item()

        allowed_units = ALLOWED_UNITS.get(entity_name, [])
        if not allowed_units:
            return ""

        unit = allowed_units[0]
        prediction = f"{value:.2f} {unit}"
        return prediction
    except Exception as e:
        print(f"Error processing {image_link}: {str(e)}")
        return ""

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    possible_paths = [
        DATASET_FOLDER,
        DRIVE_PATH,
        '/Users/shivamsharma/Desktop/Amazon-ml/dataset',
    ]

    try:
        # Load training data
        train_data = load_data('train.csv', possible_paths)
        
        # Split data into train and validation sets
        train_data, val_data = train_test_split(train_data, test_size=0.2, random_state=42)
        train_dataset = ProductImageDataset(train_data, transform=transform)
        val_dataset = ProductImageDataset(val_data, transform=transform)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=4, pin_memory=True)

        # Initialize model, loss function, and optimizer
        model = ProductFeatureExtractor(num_entity_types=len(train_dataset.entity_types)).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, verbose=True)

        # Train the model
        train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, NUM_EPOCHS, device)

        # Load the best model for prediction
        model.load_state_dict(torch.load(os.path.join(DRIVE_PATH, 'best_model.pth'), map_location=device))
        model.eval()

        # Load test data and make predictions
        try:
            test_data = load_data('test.csv', possible_paths)
        except FileNotFoundError:
            print("test.csv not found, loading sample_test.csv instead.")
            test_data = load_data('sample_test.csv', possible_paths)

        # Generate predictions
        tqdm.pandas()
        test_data['prediction'] = test_data.progress_apply(
            lambda row: predictor(model, row['image_link'], row['entity_name'], train_dataset.entity_to_idx, device),
            axis=1
        )

        # Create output directory if it doesn't exist
        output_dir = os.path.join(DRIVE_PATH, 'output')
        os.makedirs(output_dir, exist_ok=True)

        # Save predictions
        output_filename = os.path.join(output_dir, 'test_out.csv')
        test_data[['index', 'prediction']].to_csv(output_filename, index=False)
        print(f"Predictions saved to {output_filename}")

        # Run sanity check
        sanity_script_path = os.path.join(DRIVE_PATH, 'src', 'sanity.py')
        os.system(f"python {sanity_script_path} {output_filename}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please make sure the required files are in one of the following locations:")
        for path in possible_paths:
            print(f"- {path}")
        print("If the files are in a different location, please add the path to the 'possible_paths' list in the code.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        print("Please check your code and data, and try again.")

if __name__ == "__main__":
    main()