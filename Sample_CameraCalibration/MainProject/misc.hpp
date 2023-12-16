#pragma once

void checkFileExists(String filename) {
	ifstream infile;
	infile.open(filename, ios::in);
	if (infile.fail())
	{
		cout << "ERROR. Failed to open file.\n";
		return;
	}
	else {
		cout << "File opened.\n";
		infile.close();
		return;
	}
}