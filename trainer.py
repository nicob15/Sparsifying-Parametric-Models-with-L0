import torch

def train_eval_dynamics_model(model, optimizer, train_loader, test_loader, batch_size, num_epochs, l0=False):

    nr_data = train_loader.size
    iterations = int(nr_data/batch_size)

    best_train_error = 1e12
    best_eval_error = 1e12
    best_train_error_epoch = 0
    best_eval_error_epoch = 0

    train_error_list = []
    eval_error_list = []

    for epoch in range(num_epochs):
        train_loss = 0
        reg_loss = 0
        model.train()
        for i in range(iterations):
            data = train_loader.sample_batch(batch_size)

            obs = torch.from_numpy(data['obs']).cuda()
            next_obs = torch.from_numpy(data['next_obs']).cuda()
            act = torch.from_numpy(data['act']).cuda()

            optimizer.zero_grad()

            if l0:
                model.training = True
            pred_next_obs = model(obs, act)

            if l0 == False:
                loss = torch.nn.functional.mse_loss(next_obs, pred_next_obs)
                total_loss = loss
            else:

                loss = torch.nn.functional.mse_loss(next_obs, pred_next_obs)
                reg = -(model.fc.regularization() + model.fc1.regularization() + model.fc2.regularization())
                total_loss = loss + reg

            total_loss.backward()

            train_loss += total_loss.item()
            if l0 == True:
                reg_loss += reg.item()

            optimizer.step()

        train_error = train_loss / iterations
        if l0 == True:
            train_error = (train_loss - reg_loss) / iterations
        train_error_list.append(train_error)
        print('====> Epoch: {} Average train loss: {:.10f}'.format(epoch, train_error))
        if l0 == True:
            reg_error = reg_loss / iterations
            print('====> Epoch: {} Average L0 reg loss: {:.10f}'.format(epoch, reg_error))
        if train_error < best_train_error:
            best_train_error = train_error
            best_train_error_epoch = epoch

        eval_error = evaluate_dynamics_model(model, test_loader, epoch, l0)
        eval_error_list.append(eval_error)
        if eval_error < best_eval_error:
            best_eval_error = eval_error
            best_eval_error_epoch = epoch

    return best_train_error, best_train_error_epoch, best_eval_error, best_eval_error_epoch, train_error_list, eval_error_list

def evaluate_dynamics_model(model, test_loader, epoch, l0=False):

    model.eval()

    data = test_loader.get_all_samples()

    obs = torch.from_numpy(data['obs']).cuda()
    next_obs = torch.from_numpy(data['next_obs']).cuda()
    act = torch.from_numpy(data['act']).cuda()

    if l0:
        model.training = False
    pred_next_obs = model(obs, act)

    loss = torch.nn.functional.mse_loss(next_obs, pred_next_obs)

    eval_loss = loss.item()

    print('====> Epoch: {} Average eval loss: {:.10f}'.format(epoch, eval_loss))

    return eval_loss

def train_eval_reward_model(model, optimizer, train_loader, test_loader, batch_size, num_epochs, l0=False):

    nr_data = train_loader.size
    iterations = int(nr_data/batch_size)

    best_train_error = 1e12
    best_eval_error = 1e12
    best_train_error_epoch = 0
    best_eval_error_epoch = 0

    train_error_list = []
    eval_error_list = []

    for epoch in range(num_epochs):
        train_loss = 0
        reg_loss = 0
        model.train()
        for i in range(iterations):
            data = train_loader.sample_batch(batch_size)

            obs = torch.from_numpy(data['obs']).cuda()
            rew = torch.from_numpy(data['rew']).cuda().reshape(-1, 1)
            act = torch.from_numpy(data['act']).cuda()

            optimizer.zero_grad()

            if l0:
                model.training = True
            pred_rew = model(obs, act)

            if l0 == False:
                loss = torch.nn.functional.mse_loss(rew, pred_rew)
                total_loss = loss
            else:
                loss = torch.nn.functional.mse_loss(rew, pred_rew)
                reg = -(model.fc.regularization())
                total_loss = loss + reg

            total_loss.backward()

            train_loss += total_loss.item()
            if l0 == True:
                reg_loss += reg.item()

            optimizer.step()

        train_error = train_loss / iterations
        if l0 == True:
            train_error = (train_loss - reg_loss) / iterations
        train_error_list.append(train_error)
        print('====> Epoch: {} Average train loss: {:.10f}'.format(epoch, train_error))
        if l0 == True:
            reg_error = reg_loss / iterations
            print('====> Epoch: {} Average L0 reg loss: {:.10f}'.format(epoch, reg_error))
        if train_error < best_train_error:
            best_train_error = train_error
            best_train_error_epoch = epoch

        eval_error = evaluate_reward_model(model, test_loader, epoch, l0)
        eval_error_list.append(eval_error)
        if eval_error < best_eval_error:
            best_eval_error = eval_error
            best_eval_error_epoch = epoch

    return best_train_error, best_train_error_epoch, best_eval_error, best_eval_error_epoch, train_error_list, eval_error_list

def evaluate_reward_model(model, test_loader, epoch, l0=False):

    model.eval()

    data = test_loader.get_all_samples()

    obs = torch.from_numpy(data['obs']).cuda()
    rew = torch.from_numpy(data['rew']).cuda().reshape(-1, 1)
    act = torch.from_numpy(data['act']).cuda()

    if l0:
        model.training = False
    pred_rew = model(obs, act)

    loss = torch.nn.functional.mse_loss(rew, pred_rew)

    eval_loss = loss.item()

    print('====> Epoch: {} Average eval loss: {:.10f}'.format(epoch, eval_loss))

    return eval_loss
