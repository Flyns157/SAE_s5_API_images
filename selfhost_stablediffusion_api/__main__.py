from . import GenerationAPI
import argparse
import sys

# =================================== Main ===================================
def main(debug=False, host='0.0.0.0', port=8081):
    app = GenerationAPI(__name__,
                static_url_path='',
                static_folder='assets',
                template_folder='templates')
    app.run(debug=debug, host=host, port=port)

# =================================== Run ===================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Python script")
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host')
    parser.add_argument('--port', type=int, default=8081, help='Port')
    args = parser.parse_args()

    rc = 1
    try:
        main(debug=args.debug, host=args.host, port=args.port)
        rc = 0
    except Exception as e:
        print('Error: %s' % e, file=sys.stderr)
    sys.exit(rc)
