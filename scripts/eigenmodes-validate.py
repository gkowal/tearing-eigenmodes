#!/usr/bin/env python3
"""
CLI script to validate and repair/backfill .npz eigenmode files in a given path.
"""

import os
import sys
import glob
import argparse
import logging

from tearing_eigenmodes import setup_logging, validate_and_fix_file

logger = logging.getLogger("eigenmodes-validate")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate and repair/backfill tearing instability eigenmode .npz files."
    )
    parser.add_argument(
        "path",
        type=str,
        help="Path to a directory containing .npz files or to a specific .npz file."
    )
    parser.add_argument(
        "-d", "--dry-run",
        action="store_true",
        help="Validate files and report issues without updating them."
    )
    parser.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Do not traverse subdirectories recursively (default is recursive)."
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Print verbose output for each validated file."
    )
    parser.set_defaults(recursive=True)

    args = parser.parse_args()

    # Configure logging
    setup_logging(verbose=args.verbose)

    # Determine files to process
    target_path = os.path.abspath(args.path)
    if not os.path.exists(target_path):
        logger.error(f"Path does not exist: {target_path}")
        sys.exit(1)

    files = []
    if os.path.isfile(target_path):
        if target_path.endswith(".npz"):
            files.append(target_path)
        else:
            logger.error(f"Target file is not a .npz file: {target_path}")
            sys.exit(1)
    else:
        # It's a directory
        pattern = "**/*.npz" if args.recursive else "*.npz"
        search_pattern = os.path.join(target_path, pattern)
        files = sorted(glob.glob(search_pattern, recursive=args.recursive))

    if not files:
        logger.warning(f"No .npz files found in {target_path}")
        sys.exit(0)

    logger.info(f"Starting validation of {len(files)} file(s)...")
    if args.dry_run:
        logger.info("[DRY-RUN mode enabled - no files will be written]")

    scanned = 0
    passed = 0
    updated = 0
    failed = 0

    for f in files:
        scanned += 1
        success, modified = validate_and_fix_file(f, dry_run=args.dry_run, verbose=args.verbose)

        if success:
            if modified:
                updated += 1
                if args.verbose:
                    if args.dry_run:
                        logger.info(f"Needs update: {os.path.basename(f)}")
                    else:
                        logger.info(f"Updated: {os.path.basename(f)}")
            else:
                passed += 1
                if args.verbose:
                    logger.info(f"Valid: {os.path.basename(f)}")
        else:
            failed += 1
            logger.error(f"Failed validation: {os.path.basename(f)}")

    # Print summary
    logger.info("\n=== Validation Summary ===")
    logger.info(f"Total files scanned: {scanned}")
    logger.info(f"Already valid:       {passed}")
    if args.dry_run:
        logger.info(f"Needs update:        {updated}")
    else:
        logger.info(f"Successfully updated:{updated}")
    logger.info(f"Invalid / Corrupted: {failed}")

    if failed > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
