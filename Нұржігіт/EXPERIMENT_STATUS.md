# Current experiment

v12 code bd2b49a adds zero-gradient row removal in the frozen MLP backward.
The full forward, token order, loss, dropout and task settings remain unchanged.
Validation is pending; do not treat this snapshot as an accepted improvement.

v11 9e104d4: frozen output-head cast cache and sparse backward, queued for validation.
v10 e19b464: 54 kernel tests and 79 Soup tests passed; paired validation is running.
The pinned candidate notebook still runs validated v9 (57182a2).
