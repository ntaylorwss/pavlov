import getpass
import hashlib
import random

# get password input
password = None
if password is None:
    for attempt in range(3):
        p0 = getpass.getpass('Enter password for jupyter notebook: ')
        p1 = getpass.getpass('Verify password: ')
        if p0 == p1:
            password = p0
            break
        else:
            print('Passwords do not match.')
    else:
        raise ValueError('No matching passwords found. Giving up.')

# hash password; straight up stolen from jupyter/notebook/auth/security.py/passwd()
h = hashlib.new('sha1')
salt_len = 12
salt = ('%0' + str(salt_len) + 'x') % random.getrandbits(4 * salt_len)
h.update(password.encode('utf-8') + salt.encode('ascii'))
passphrase = ':'.join(('sha1', salt, h.hexdigest()))

with open('jupyter_notebook_config.json', 'w') as f:
    full_json = '{{"NotebookApp": {{"password": "{}"}}}}'.format(passphrase)
    f.write(full_json)
