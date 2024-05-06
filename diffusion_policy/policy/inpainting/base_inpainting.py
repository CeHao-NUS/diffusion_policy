
import torch
import numpy as np
import cvxpy as cp



class BaseInpainting:


    # parse observation
    def inpaint(self):
        pass


    # create mask and action data
    def create_mask_and_data(self):
        pass

    def set_mask_cond(self, mask, cond):
        self.mask = mask
        self.cond = cond

def base_inpaint(x, mask, cond, **kwargs):
    pass


# vanilla inpainting
def inpaint_vanilla(x, mask, cond):
    x_inpaint = mask * x + (1 - mask) * cond
    return x_inpaint


# inpainting opt
def inpaint_opt():
    pass


def MSE_inequ_opt(x, mask, patch, threshold):


    # x (B, length, dim)

    batch_size = x.shape[0]
    action_length = x.shape[1]
    num_non_zero = np.sum(mask) # number of non zero elements in mask


    # flatten the second and third dimensions
    x_flat = x[0]
    mask_flat = mask[0]
    patch_flat = patch[0]

    # create optimization variable
    x_hat = cp.Variable(x_flat.shape)
    x_hat.value = x_flat

    # create optimization problem
    # loss 1: inpainting loss
    masked_diff = cp.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = cp.sum_squares(masked_diff) / num_non_zero / batch_size

    # loss 2: in-dist loss
    backward_diff = x_hat - x_flat
    mse_loss_backward = cp.sum_squares(backward_diff)/ action_length / batch_size

    # total loss with weight
    total_loss = mse_loss_mask 

    # inequality constraint, mse_loss_backward < threshold
    constraint = [mse_loss_backward <= threshold]

    # create optimization problem and solve
    problem = cp.Problem(cp.Minimize(total_loss), constraints=constraint)

    problem.solve()

    # get optimized results
    x_hat = x_hat.value


    '''
    masked_diff = np.multiply(mask_flat, (x_hat - patch_flat))
    mse_loss_mask = np.mean(masked_diff**2) / num_non_zero
    backward_diff = x_hat - x_flat
    mse_loss_backward = np.mean(backward_diff**2) / action_length

    print(f'loss patch: {mse_loss_mask}')
    print(f'loss in-dist: {mse_loss_backward}')
    
    '''

    # reshape to original shape
    x_hat = x_hat.reshape(x.shape)


    return x_hat