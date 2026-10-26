### Commom parameters

| Variable                                | Type    | Description                                                                                               |
|-----------------------------------------|---------|-----------------------------------------------------------------------------------------------------------|
| user_type                               | string  | Type of user, reffers to the name of the conf file itself and the behavior of the user during simulation. |
|                                         |         |                                                                                                           |
| user_idx                                | int     | The id of which one of the clients in the simulation that will be attacked.                               |
|                                         |         |                                                                                                           |
| num_data_points                         | int     | Size of the batch that the user will train on simulation.                                                 |
|                                         |         |                                                                                                           |
| provide_buffers                         | boolean | Defines if user will provide buffers to server during simulation.                                         |
|                                         |         |                                                                                                           |
| provide_labels                          | boolean | Defines if user will provide labels to server during simulation.                                          |
|                                         |         |                                                                                                           |
| provide_num_data_points                 | boolean | Defines if user will provide the size of trained data to server during simulation.                        |
|                                         |         |                                                                                                           |
| local_diff_privacy                      | object  | Defines the values of local differencial privacy used by the stragety during simulation.                  |
|                                         |         |                                                                                                           |
| local_diff_privacy.gradient_noise       | float   |                                                                                                           |
|                                         |         |                                                                                                           |
| local_diff_privacy.input_noise          | float   |                                                                                                           |
|                                         |         |                                                                                                           |
| local_diff_privacy.distribution         | enum    | Can be the following values:                                                                              |
|                                         |         | gaussian                                                                                                  |
|                                         |         | laplacian                                                                                                 |
| local_diff_privacy.per_example_clipping | float   |                                                                                                           |

### Parameters specific to local_updates and multiuser_aggregate

| Variable                       | Type    | Description                                                               |
|--------------------------------|---------|---------------------------------------------------------------------------|
| num_local_updates              | int     | Number of rounds before sending the gradient to server.                   |
|                                |         |                                                                           |
| num_data_per_local_update_step | int     | Number of trained data before sending the gradient to server.             |
|                                |         |                                                                           |
| local_learning_rate            | float   | Learning rate optimizer used while training.                              |
|                                |         |                                                                           |
| provide_local_hyperparams      | boolean | Defines if client will share those parameters with server on update time. |


