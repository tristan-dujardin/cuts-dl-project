import argparse
from dataset.dataset_utils import split_data
from dataset.retina import Retina
from model.model import CUTSModel, train, test

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parser for Model Settings.')
    parser.add_argument('--mode', type=str, default='test', help='`train` or `test` the model')
    parser.add_argument('--lr', type=float, default=0.03, help='Learning Rate float, unused if --mode is `test`')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs int, unused if --mode is `test`')

    params = parser.parse_args()

    data = Retina()

    num_image_channels = data.num_image_channel()
    print(f'nic: {num_image_channels}')
    train_loader, val_loader, test_loader = split_data(data, (0.7, 0.15, 0.15), 8, max_size=200)

    print('Creating Model')
    model = CUTSModel(num_image_channels)
    print('Created model')

    model = train(model, train_loader, val_loader, params, verbose=True)
    print('Trained model')

    loss = test(model, test_loader)

    print(f'Run Test Loss: {loss}\n')

    save = input('Will you Save the model? [y/N]\n').lower()

    if save in ['y', 'yes']:
        model.save_model('models/weights.pth')