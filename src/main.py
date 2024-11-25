

from model.model import CUTSModel, train, test

if __name__ == '__main__':
    params = {'learning_rate': 0.03}
    model = CUTSModel()



    train_loader = None
    val_loader = None
    test_loader = None

    model = train(model, train_loader, val_loader, params, verbose=True)

    acc = test(model, test_loader)

    print(f'Run Test Accuracy: {acc:.2%}\n')

    save = input('Will you Save the model? [y/N]\n').lower()

    if save in ['y', 'yes']:
        model.save_model('models/weights.pth')