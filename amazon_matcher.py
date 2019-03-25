import boto3
from os import listdir, path
import argparse


def match_faces(probe_path, gallery_path):
    probe = listdir(probe_path)
    gallery = listdir(gallery_path)

    client = boto3.client('rekognition')

    for i in range(len(probe)):

        if gallery_path != probe_path:
            i = -1

        for j in range(i + 1, len(gallery)):
            sourceFile = path.join(probe_path, probe[i])
            targetFile = path.join(gallery_path, gallery[j])

            imageTarget = open(sourceFile, 'rb')
            imageSource = open(targetFile, 'rb')

            try:
                response = client.compare_faces(SimilarityThreshold=0,
                                                SourceImage={'Bytes': imageSource.read()},
                                                TargetImage={'Bytes': imageTarget.read()})

                for faceMatch in response['FaceMatches']:
                    similarity = str(faceMatch['Similarity'])
                    quality = faceMatch['Face']['Quality']

                    print('Similarity between {0} and {1}: {2}'.
                          format(probe[i], gallery[j], similarity))
                    print('Targer quality: {}'.format(quality))
                    print()

            except:
                print('Error trying to match faecs {0} and {1}'.format(
                    probe[i], gallery[j]))

            imageSource.close()
            imageTarget.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Amazon matcher')
    parser.add_argument('-probe', '-p', help='Probe image list.')
    parser.add_argument('-gallery', '-g', help='Gallery image list.')

    args = parser.parse_args()

    if args.gallery is None:
        args.gallery = args.probe

    match_faces(args.probe, args.gallery)
