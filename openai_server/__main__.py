import sys
import os
from . import app

if __name__ == '__main__':
  args = sys.argv[1:]
  port = int(args[0] if len(args) > 0 else os.environ.get('PORT', '9000'))
  app.run(host='0.0.0.0', port=port)

