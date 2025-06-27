#!/usr/bin/env python
"""Django's command-line utility for administrative tasks."""
import os
import sys

def main():
    """Run administrative tasks."""
    # ✨ 백그라운드 스레드 시작 로직 추가 ✨
    # runserver시에만 스레드가 실행되도록 조건 추가
    if 'runserver' in sys.argv:
        from api.data_manager import start_data_workers
        print("Starting data worker threads...")
        start_data_workers()
    # ---

    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'config.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed and "
            "available on your PYTHONPATH environment variable? Did you "
            "forget to activate a virtual environment?"
        ) from exc
    execute_from_command_line(sys.argv)


if __name__ == '__main__':
    main()