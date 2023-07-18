#pragma once
#include "image2d.h"
#include "image3d.h"
#include <algorithm>
#include <numeric>
#include <fstream>



enum {
	MRC_NUM_LABELS = 10,
	MRC_LABEL_SIZE = 80,
	NUM_4BYTES_PRE_MAP = 52,
	NUM_4BYTES_AFTER_MAP = 3
};

struct MRCheader {
	int nx;                 /* number of columns */
	int ny;                 /* number of rows */
	int nz;                 /* number of sections */

	/*
	Types of pixel in image.Values used by IMOD :
	0 = unsigned or signed bytes depending on flag in imodStamp
	1 = signed short integers(16 bits)
	2 = float
	3 = short * 2, (used for complex data)
	4 = float * 2, (used for complex data)
	6 = unsigned 16 - bit integers
	16 = unsigned char * 3 (for rgb data, non - standard)
	101 = 4 - bit values(non - standard)
	*/
	int mode;               /* See modes above. */

	int nxstart;            /* No. of first column in map, default 0. */
	int nystart;            /* No. of first row in map, default 0. */
	int nzstart;            /* No. of first section in map,default 0. */

	int mx;                 /* Number of intervals along X. */
	int my;                 /* Number of intervals along Y. */
	int mz;                 /* Number of intervals along Z. */

							/* Cell: treat a whole 2D image as a cell */
	float xlen;             /* Cell dimensions (Angstroms). */
	float ylen;             /* Cell dimensions (Angstroms). */
	float zlen;             /* Cell dimensions (Angstroms). */

	float alpha;            /* Cell angles (Degrees). */
	float beta;             /* Cell angles (Degrees). */
	float gamma;            /* Cell angles (Degrees). */

							/* axis X => 1, Y => 2, Z => 3 */
	int mapc;               /* Which axis corresponds to Columns.  */
	int mapr;               /* Which axis corresponds to Rows.     */
	int maps;               /* Which axis corresponds to Sections. */
	float amin;             /* Minimum density value. */
	float amax;             /* Maximum density value. */
	float amean;            /* Mean density value.    */

	int ispg;               /* Space group number (0 for images). */

	int nsymbt;             /* Number of chars used for storing symmetry */
							/* operators.                                */

	int user[25];

	float xorigin;          /* X origin. */
	float yorigin;          /* Y origin. */
	float zorigin;          /* Y origin. */

	char map[4];            /* constant string "MAP "  */
	int machinestamp;       /* machine stamp in CCP4 convention: */
							/* big endian=0x11110000 little endian=0x4444000 */

	float rms;              /* rms deviation of map from mean density */

	int nlabels;            /* Number of labels being used. */
	char labels[MRC_NUM_LABELS][MRC_LABEL_SIZE];
};

inline  MRCheader mrcheader_from_image(image3d<float> img)
{
	MRCheader mrcheader;

	mrcheader.nx = static_cast<int>(img.nx());
	mrcheader.ny = static_cast<int>(img.ny());
	mrcheader.nz = static_cast<int>(img.nz());
	mrcheader.mode = 2;
	mrcheader.nxstart = 0;
	mrcheader.nystart = 0;
	mrcheader.nzstart = 0;
	mrcheader.mx = static_cast<int>(img.nx());
	mrcheader.my = static_cast<int>(img.ny());
	mrcheader.mz = static_cast<int>(img.nz());

	mrcheader.xlen = 1;   /* Cell dimensions (Angstroms). */
	mrcheader.ylen = 1;   /* Cell dimensions (Angstroms). */
	mrcheader.zlen = 1;   /* Cell dimensions (Angstroms). */

	mrcheader.alpha = 90.0;  /* Cell angles (Degrees). */
	mrcheader.beta = 90.0;   /* Cell angles (Degrees). */
	mrcheader.beta = 90.0;   /* Cell angles (Degrees). */
	mrcheader.gamma = 90.0;  /* Cell angles (Degrees). */

	mrcheader.mapc = 1;  /* Which axis corresponds to Columns.  */
	mrcheader.mapr = 2;  /* Which axis corresponds to Rows.     */
	mrcheader.maps = 3;

	mrcheader.amin = static_cast<float>(*std::min_element(
		img.m_data.begin(), img.m_data.end()));
	mrcheader.amax = static_cast<float>(*std::max_element(
		img.m_data.begin(), img.m_data.end()));
	mrcheader.amean = static_cast<float>(std::accumulate(
		img.m_data.begin(), img.m_data.end(), 0.0) / img.m_data.size());

	mrcheader.ispg = 0; /* Space group number (0 for images). */
	mrcheader.nsymbt = 0; /* Number of chars used for storing symmetry */
						  /* operators.                                */
	for (int i = 0; i<25; i++) mrcheader.user[i] = ' ';
	mrcheader.user[24] = '\0';

	mrcheader.xorigin = 0;
	mrcheader.yorigin = 0;
	mrcheader.zorigin = 0;

	const int n = 1;
	if (*(char*)&n == 1) mrcheader.machinestamp = 0x4144;
	else mrcheader.machinestamp = 0x1111;

	mrcheader.map[0] = 'M';
	mrcheader.map[0] = 'M';
	mrcheader.map[1] = 'A';
	mrcheader.map[2] = 'P';
	mrcheader.map[3] = '\0';

	mrcheader.rms = 1.0;
	mrcheader.nlabels = 0;

	return mrcheader;
};

#define _CRT_SECURE_NO_WARNINGS
#pragma warning(disable:4996)
inline bool writeImage(image3d<float> img, std::string filename)
{
	const MRCheader mrcheader = mrcheader_from_image(img);

	bool status = true;

	FILE *fp;
	size_t imagesize = img.size();

	fp = fopen(filename.c_str(), "wb");
	if (!fp) {
		fprintf(stderr, "failed to open %s\n", filename.c_str());
		status = false;
		goto EXIT;
	}
	if (fwrite(&mrcheader, sizeof(MRCheader), 1, fp) != 1) {
		fprintf(stderr, "failed to write an MRC header\n");
		status = false;
		goto EXIT;
	}

	if (fwrite(img.m_data.data(), sizeof(float), imagesize, fp) != imagesize) {
		fprintf(stderr, "failed to write MRC data\n");
		status = false;
		fclose(fp);
		goto EXIT;
	}
	fclose(fp);
EXIT:
	return status;

}

inline image3d<float> arecReadImage_stream(std::string filename)
{
	std::ifstream inStream(filename, std::ios::binary);
	if (inStream.is_open())
	{	
		
		MRCheader mrcheader;
		inStream.read(reinterpret_cast<char*>(&mrcheader), sizeof(mrcheader));

		int nx = mrcheader.nx;
		int ny = mrcheader.ny;
		int nz = mrcheader.nz;
		const int mode = mrcheader.mode;
		printf("Image of size (%d,%d,%d) \n", nx, ny, nz);
		printf("Endian stamp: stamp 0x%X  (BE=0x1111 LE=0x4144)\n", mrcheader.machinestamp);

		if (mode == 2) {
			image3d<float> img = image3d<float>(nx, ny, nz);
			inStream.read(reinterpret_cast<char*>(img.data()), img.size() * sizeof(float));
			return img;
		}
		else {
			fprintf(stderr, "arecReadImage: invalid mrc mode: %d\n", mode);
		}
		
	}
	else
	{
		fprintf(stderr, "Unable to open file \n");
	}
	return image3d<float>();
}

inline  void writeImage_stream(image3d<float> img, std::string filename)
{
	const MRCheader mrcheader = mrcheader_from_image(img);
	std::ofstream outStream(filename, std::ios::binary);

	outStream.write(reinterpret_cast<const char*>(&mrcheader), sizeof(mrcheader));
	outStream.write(reinterpret_cast<char*>(img.data()), img.size() * sizeof(float));

}


