import sys
import argparse
import logging
import pathlib
import csv
import cv2
from blur_detection import estimate_blur, fix_image_size, pretty_blur_map, extract_text


def parse_args():
    parser = argparse.ArgumentParser(description='Run blur detection and OCR on images')
    parser.add_argument('-i', '--images', type=str, nargs='+', required=True, help='Paths to images or directories')
    parser.add_argument('-s', '--save-path', type=str, default=None, help='Path to save the CSV output')
    parser.add_argument('-t', '--threshold', type=float, default=45.0, help='Blurry threshold (lower is more sensitive)')
    parser.add_argument('-o', '--ocr-threshold', type=float, default=50.0, help='Minimum OCR confidence to consider text readable')
    parser.add_argument('-f', '--variable-size', action='store_true', help='Fix the image size for consistent scoring')
    parser.add_argument('-v', '--verbose', action='store_true', help='Set logging level to debug')
    parser.add_argument('-d', '--display', action='store_true', help='Display images during processing')

    return parser.parse_args()


def find_images(image_paths, img_extensions=['.jpg', '.png', '.jpeg']):
    img_extensions += [i.upper() for i in img_extensions]

    for path in image_paths:
        path = pathlib.Path(path)

        if path.is_file():
            if path.suffix not in img_extensions:
                logging.info(f'{path.suffix} is not an image extension! Skipping {path}')
                continue
            else:
                yield path

        if path.is_dir():
            for img_ext in img_extensions:
                yield from path.rglob(f'*{img_ext}')


if __name__ == '__main__':
    assert sys.version_info >= (3, 6), sys.version_info
    args = parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level)

    fix_size = not args.variable_size
    logging.info(f'fix_size: {fix_size}')

    if args.save_path is not None:
        save_path = pathlib.Path(args.save_path)
        assert save_path.suffix == '.csv', "Output file must be in .csv format"
    else:
        save_path = pathlib.Path("blur_results.csv")

    results = []

    for image_path in find_images(args.images):
        image = cv2.imread(str(image_path))
        if image is None:
            logging.warning(f'Warning! Failed to read image from {image_path}; skipping!')
            continue

        logging.info(f'Processing {image_path}')

        if fix_size:
            image = fix_image_size(image)
        else:
            logging.warning('Not normalizing image size for consistent scoring!')

        # Estimate blur
        blur_map, blur_score, is_blurry = estimate_blur(image, threshold=args.threshold)

        # Perform OCR
        extracted_text, ocr_confidence = extract_text(image)

        # Determine final blurriness based on OCR and sharpness
        is_text_clear = ocr_confidence >= args.ocr_threshold
        final_blurry = is_blurry or not is_text_clear
        prediction = "Yes" if final_blurry else "No"

        logging.info(f'Image: {image_path} | Blur Score: {blur_score} | OCR Confidence: {ocr_confidence} | Blurry: {prediction}')

        results.append([image_path.name, prediction])

        if args.display:
            cv2.imshow('Input', image)
            cv2.imshow('Blur Map', pretty_blur_map(blur_map))

            if cv2.waitKey(0) == ord('q'):
                logging.info('Exiting...')
                exit()

    # Save results to CSV
    logging.info(f'Saving CSV results to {save_path}')

    try:
        with open(save_path, 'w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["Image Name", "Prediction (Blurry: Yes/No)"])
            writer.writerows(results)

        logging.info(f"Results successfully saved in {save_path}")

    except Exception as e:
        logging.error(f"Failed to save results: {e}")
