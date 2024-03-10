#include <stdio.h>
#include <assert.h>
#include <stdlib.h> 
#include <stdbool.h>
#include <time.h>

//#define DEBUG
#define PROGRESS
//#define BITCOUNT

typedef struct {
	FILE* weight_ptr;
	FILE* bias_ptr;
	FILE* ifmaps_ptr;
	FILE* psum_before_bias_ptr;
	FILE* psum_after_bias_ptr;
	FILE* ofmaps_ptr;
	FILE* setting_ptr;
	int function;
	int ifmaps_width;
	int ifmaps_hight;
	int ifmaps_channel;
	int weight_width;
	int weight_hight;
	int weight_num;
	int stride;
}data_setting;

void write_file_setup();
void using_user_data_setup();
data_setting check_file(int wr_ofmaps);
void write_conv(data_setting* setting);
void write_pool(data_setting* setting);
bool*** construct_bool3Darray(int z, int y, int x);
bool**** construct_bool4Darray(int w, int z, int y, int x);
int*** construct_int3Darray(int z, int y, int x);
int* construct_int1Darray(int x);


int main()
{
	printf("請輸入模式 \n9 為寫入檔案 \n8 為讀取ifmaps weight setting檔案計算 \n其餘為讀取setting檔案並驗證檔案\n");
	int mode = 0;
	scanf_s("%d", &mode);
	if (mode == 9)
	{
		write_file_setup();
	}
	else if (mode == 8)
	{
		using_user_data_setup();
	}
	else
	{
		check_file(0);
	}
	system("pause");
	return 0;
}

void write_file_setup()
{
	data_setting wf_setting = {NULL,NULL,NULL,NULL,NULL,NULL,NULL,0,0,0,0,0,0,0,0 };

	printf("使否增加檔案index? 1為需要其餘為不需要\n");
	int add_index = 0;
	scanf_s("%d", &add_index);
	char index[2]="";
	if (add_index == 1) {
		printf("請輸入檔案index\n");
		scanf_s("%s", index, 2);
	}

	char file_dir[50];

	sprintf(file_dir, "../weight%s.txt", index);
	if ((wf_setting.weight_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("找不到weight.txt將自動產生檔案");
	}

	sprintf(file_dir, "../bias%s.txt", index);
	if ((wf_setting.bias_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("找不到bias.txt將自動產生檔案");
	}

	sprintf(file_dir, "../ifmaps%s.txt", index);
	if ((wf_setting.ifmaps_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("找不到ifmaps.txt將自動產生檔案");
	}

	sprintf(file_dir, "../psum_before_bias%s.txt", index);
	if ((wf_setting.psum_before_bias_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("找不到psum_before_bias.txt將自動產生檔案");
	}

	sprintf(file_dir, "../psum_after_bias%s.txt", index);
	if ((wf_setting.psum_after_bias_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("找不到psum_after_bias.txt將自動產生檔案");
	}

	sprintf(file_dir, "../ofmaps%s.txt", index);
	if ((wf_setting.ofmaps_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("找不到ofmaps.txt將自動產生檔案");
	}

	sprintf(file_dir, "../setting%s.txt", index);
	if ((wf_setting.setting_ptr = fopen(file_dir, "wb+")) == NULL)
	{
		printf("setting.txt將自動產生檔案");
	}

	printf("請輸入執行的function 0為convolution 1 為pooling\n");
	scanf_s("%d", &wf_setting.function);
	fprintf(wf_setting.setting_ptr, "//function\n%d\n", wf_setting.function);

	assert(wf_setting.function == 0 || wf_setting.function == 1);

	if (wf_setting.function == 0)
	{
		printf("請輸入ifmaps寬度");
		scanf_s("%d", &wf_setting.ifmaps_width);
		fprintf(wf_setting.setting_ptr, "//ifmaps_width\n%d\n", wf_setting.ifmaps_width);
		printf("請輸入ifmaps高度");
		scanf_s("%d", &wf_setting.ifmaps_hight);
		fprintf(wf_setting.setting_ptr, "//ifmaps_hight\n%d\n", wf_setting.ifmaps_hight);
		printf("請輸入ifmaps通道數");
		scanf_s("%d", &wf_setting.ifmaps_channel);
		fprintf(wf_setting.setting_ptr, "//ifmaps_channel\n%d\n", wf_setting.ifmaps_channel);

		printf("請輸入weight寬度");
		scanf_s("%d", &wf_setting.weight_width);
		fprintf(wf_setting.setting_ptr, "//weight_width\n%d\n", wf_setting.weight_width);
		printf("請輸入weight高度");
		scanf_s("%d", &wf_setting.weight_hight);
		fprintf(wf_setting.setting_ptr, "//weight_hight\n%d\n", wf_setting.weight_hight);
		printf("請輸入weight數量");
		scanf_s("%d", &wf_setting.weight_num);
		fprintf(wf_setting.setting_ptr, "//weight_num\n%d\n", wf_setting.weight_num);

		printf("請輸入步距stride");
		scanf_s("%d", &wf_setting.stride);
		fprintf(wf_setting.setting_ptr, "//stride\n%d\n", wf_setting.stride);

		write_conv(&wf_setting);
	}
	else if (wf_setting.function == 1)
	{
		printf("請輸入ifmaps寬度");
		scanf_s("%d", &wf_setting.ifmaps_width);
		fprintf(wf_setting.setting_ptr, "//ifmaps_width\n%d\n", wf_setting.ifmaps_width);
		printf("請輸入ifmaps高度");
		scanf_s("%d", &wf_setting.ifmaps_hight);
		fprintf(wf_setting.setting_ptr, "//ifmaps_hight\n%d\n", wf_setting.ifmaps_hight);
		printf("請輸入ifmaps通道數");
		scanf_s("%d", &wf_setting.ifmaps_channel);
		fprintf(wf_setting.setting_ptr, "//ifmaps_channel\n%d\n", wf_setting.ifmaps_channel);

		printf("請輸入weight寬度");
		scanf_s("%d", &wf_setting.weight_width);
		fprintf(wf_setting.setting_ptr, "//weight_width\n%d\n", wf_setting.weight_width);
		printf("請輸入weight高度");
		scanf_s("%d", &wf_setting.weight_hight);
		fprintf(wf_setting.setting_ptr, "//weight_hight\n%d\n", wf_setting.weight_hight);
		wf_setting.weight_num = wf_setting.ifmaps_channel;
		fprintf(wf_setting.setting_ptr, "//weight_num\n%d\n", wf_setting.weight_num);

		printf("請輸入步距stride");
		scanf_s("%d", &wf_setting.stride);
		fprintf(wf_setting.setting_ptr, "//stride\n%d\n", wf_setting.stride);

		write_pool(&wf_setting);

	}
}

void using_user_data_setup()
{
	data_setting setting;

	bool*** ifmaps;
	bool**** weight;
	int* bias;
	int*** psum_before_bias;
	int*** psum_after_bias;
	bool*** ofmaps;

	setting = check_file(1);

	int ofmaps_width = ((setting.ifmaps_width - setting.weight_width) / setting.stride + 1);
	int ofmaps_hight = ((setting.ifmaps_hight - setting.weight_hight) / setting.stride + 1);

	ifmaps = construct_bool3Darray(setting.ifmaps_channel, setting.ifmaps_hight, setting.ifmaps_width);
	weight = construct_bool4Darray(setting.weight_num, setting.ifmaps_channel, setting.ifmaps_hight, setting.ifmaps_width);
	bias = construct_int1Darray(setting.weight_num);
	psum_before_bias = construct_int3Darray(setting.weight_num, ofmaps_hight, ofmaps_width);
	psum_after_bias = construct_int3Darray(setting.weight_num, ofmaps_hight, ofmaps_width);
	ofmaps = construct_bool3Darray(setting.weight_num, ofmaps_hight, ofmaps_width);


	for (int ifmaps_ch = 0; ifmaps_ch < setting.ifmaps_channel; ifmaps_ch++)
	{
		for (int ifmaps_h = 0; ifmaps_h < setting.ifmaps_hight; ifmaps_h++)
		{
			for (int ifmaps_w = 0; ifmaps_w < setting.ifmaps_width; ifmaps_w++)
			{
				fscanf(setting.ifmaps_ptr, "%d", &ifmaps[ifmaps_ch][ifmaps_h][ifmaps_w]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
#ifdef DEBUG
				printf("%d ", ifmaps[ifmaps_ch][ifmaps_h][ifmaps_w]);
#endif // DEBUG

			}
			fscanf(setting.ifmaps_ptr, "\n");
#ifdef DEBUG
			printf("\n");
#endif // DEBUG

		}
		fscanf(setting.ifmaps_ptr, "\n");
#ifdef DEBUG
		printf("\n");
#endif // DEBUG

	}
	fclose(setting.ifmaps_ptr);

	for (int weight_num = 0; weight_num < setting.weight_num; weight_num++)
	{
		for (int ifmaps_ch = 0; ifmaps_ch < setting.ifmaps_channel; ifmaps_ch++)
		{
			for (int weight_h = 0; weight_h < setting.weight_hight; weight_h++)
			{
				for (int weight_w = 0; weight_w < setting.weight_width; weight_w++)
				{
					fscanf(setting.weight_ptr, "%d", &weight[weight_num][ifmaps_ch][weight_h][weight_w]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
#ifdef DEBUG
					printf("%d ", weight[weight_num][ifmaps_ch][weight_h][weight_w]);
#endif // DEBUG
				}
				fscanf(setting.weight_ptr, "\n");
#ifdef DEBUG
				printf("\n");
#endif // DEBUG
			}
			fscanf(setting.weight_ptr, "\n");
#ifdef DEBUG
			printf("\n");
#endif // DEBUG
		}
	}
	fclose(setting.weight_ptr);

	for (int weight_num = 0; weight_num < setting.weight_num; weight_num++)
	{
		fscanf(setting.bias_ptr, "%d", &bias[weight_num]);/*每次讀取一個數，fscanf函式遇到空格或者換行結束*/
		//printf("%d ", bias[weight_num]);
	}
	fclose(setting.bias_ptr);


	for (int o_ch = 0; o_ch < setting.weight_num; o_ch++)
	{
		for (int o_h = 0; o_h < ofmaps_hight; o_h++)
		{
			for (int o_w = 0; o_w < ofmaps_width; o_w++)
			{
				int psum = 0;
				for (int w_h = 0; w_h < setting.weight_hight; w_h++)
				{
					for (int w_w = 0; w_w < setting.weight_width; w_w++)
					{
						int h = o_h * setting.stride + w_h;
						int w = o_w * setting.stride + w_w;
						if (h >= 0 && w >= 0 && h < setting.ifmaps_hight && w < setting.ifmaps_width)
						{
							for (int ch = 0; ch < setting.ifmaps_channel; ch++)
							{
#ifdef BITCOUNT
								int tmp = (!(bool)(ifmaps[ch][h][w] ^ weight[o_ch][ch][w_h][w_w])) ? 1 : 0;
#else // bitcount
								int tmp = (!(bool)(ifmaps[ch][h][w] ^ weight[o_ch][ch][w_h][w_w])) ? 1 : -1;
#endif									
								psum += tmp;
							}
						}
					}
				}
				psum_before_bias[o_ch][o_h][o_w] = psum;
				psum_after_bias[o_ch][o_h][o_w] = psum + bias[o_ch];
				fprintf(setting.psum_before_bias_ptr, "%d ", psum);
				fprintf(setting.psum_after_bias_ptr, "%d ", psum + bias[o_ch]);

#ifdef BITCOUNT
				psum = psum >= (setting.weight_hight * setting.weight_width * setting.ifmaps_channel / 2) ? 1 : 0;
#else // bitcount
				psum = psum >= 0 ? 1 : 0;
#endif
				ofmaps[o_ch][o_h][o_w] = psum;
				fprintf(setting.ofmaps_ptr, "%d ", psum);

			}
			fprintf(setting.ofmaps_ptr, "\n");
			fprintf(setting.psum_before_bias_ptr, "\n");
			fprintf(setting.psum_after_bias_ptr, "\n");
		}
		fprintf(setting.ofmaps_ptr, "\n");
		fprintf(setting.psum_before_bias_ptr, "\n");
		fprintf(setting.psum_after_bias_ptr, "\n");
#ifdef PROGRESS
		printf("進度完成第 %d 個weight計算(%d/%d)\n", (o_ch + 1), (o_ch + 1), setting.weight_num);
#endif

	}
	fclose(setting.ofmaps_ptr);
	fclose(setting.psum_before_bias_ptr);
	fclose(setting.psum_after_bias_ptr);

#ifdef DEBUG
	bool S_AXIS_TDATA[32] = { 0 };
	int cnt = 0;
	for (int x = 0; x < setting.weight_num; x = x + 1) {
		for (int y = 0; y < setting.weight_width; y = y + 1) {
			for (int z = 0; z < setting.ifmaps_channel; z = z + 6) {
				S_AXIS_TDATA[30] = 0;
				S_AXIS_TDATA[31] = 0;
				for (int w = 0; w < 30; w = w + 1) {
					if (((w / 5) + z) < setting.ifmaps_channel) {
						if ((w % 5) >= setting.weight_hight) {
							S_AXIS_TDATA[w] = 0;
						}
						else {
							S_AXIS_TDATA[w] = weight[x][(w / 5) + z][(w % 5)][y];
						}
					}
					else {
						S_AXIS_TDATA[w] = 0;
					}

				}
				for (int a = 31; a >= 0; a--)
				{
					printf("%d", S_AXIS_TDATA[a]);
				}
				printf("\n");
			}
		}
		printf("\n");
	}
	printf("\n");
	printf("\n");
	for (int x = 0; x < ofmaps_hight; x = x + 1)
	{
		for (int y = 0; y < setting.ifmaps_width; y = y + 1)
		{
			for (int z = 0; z < setting.ifmaps_channel; z = z + 6)
			{
				S_AXIS_TDATA[30] = 0;
				S_AXIS_TDATA[31] = 0;
				for (int w = 0; w < 30; w = w + 1) {
					if (((w / 5) * ((z / 6) + 1)) < setting.ifmaps_channel) {
						if (((w / 5) + z) >= setting.ifmaps_channel) {
							S_AXIS_TDATA[w] = 0;
						}
						else {
							S_AXIS_TDATA[w] = ifmaps[(w / 5) + z][w % 5 + x][y];
						}
					}
					else {
						S_AXIS_TDATA[w] = 0;
					}

				}
				for (int a = 31; a >= 0; a--)
				{
					printf("%d", S_AXIS_TDATA[a]);
				}
				cnt++;
				printf("\n");
			}

		}
		printf("\n");
	}
	printf("%d", cnt);
#endif // DEBUG

	
}

void write_conv(data_setting* setting)
{
	int bias_en = 0;
	printf("是否啟用bias 1 為啟用其餘不啟用");
	scanf_s("%d", &bias_en);

	bool*** ifmaps;
	bool**** weight;
	int* bias;
	int*** psum_before_bias;
	int*** psum_after_bias;
	bool*** ofmaps;

	int random = 0;
	int ofmaps_width = ((setting->ifmaps_width - setting->weight_width) / setting->stride + 1);
	int ofmaps_hight = ((setting->ifmaps_hight - setting->weight_hight) / setting->stride + 1);

	ifmaps = construct_bool3Darray(setting->ifmaps_channel, setting->ifmaps_hight, setting->ifmaps_width);
	weight = construct_bool4Darray(setting->weight_num, setting->ifmaps_channel, setting->ifmaps_hight, setting->ifmaps_width);
	bias = construct_int1Darray(setting->weight_num);
	psum_before_bias = construct_int3Darray(setting->weight_num, ofmaps_hight, ofmaps_width);
	psum_after_bias = construct_int3Darray(setting->weight_num, ofmaps_hight, ofmaps_width);
	ofmaps = construct_bool3Darray(setting->weight_num, ofmaps_hight, ofmaps_width);

	srand(time(NULL));

	/////////////////////////////////build data////////////////////////////////////

	for (int ifmaps_c = 0; ifmaps_c < setting->ifmaps_channel; ifmaps_c++)
	{
		for (int ifmaps_h = 0; ifmaps_h < setting->ifmaps_hight; ifmaps_h++)
		{
			for (int ifmaps_w = 0; ifmaps_w < setting->ifmaps_width; ifmaps_w++)
			{
				random = rand() % 2;
				fprintf(setting->ifmaps_ptr, "%d ", random);
				ifmaps[ifmaps_c][ifmaps_h][ifmaps_w] = random;
#ifdef DEBUG
				printf("%d", ifmaps[ifmaps_c][ifmaps_h][ifmaps_w]);
#endif // DEBUG
			}
			fprintf(setting->ifmaps_ptr, "\n");
#ifdef DEBUG
			printf("\n");
#endif // DEBUG
		}
		fprintf(setting->ifmaps_ptr, "\n");
#ifdef DEBUG
		printf("\n");
#endif // DEBUG
	}

	for (int weight_c = 0; weight_c < setting->weight_num; weight_c++)
	{
		for (int ifmaps_c = 0; ifmaps_c < setting->ifmaps_channel; ifmaps_c++)
		{
			for (int weight_h = 0; weight_h < setting->weight_hight; weight_h++)
			{
				for (int weight_w = 0; weight_w < setting->weight_width; weight_w++)
				{
					random = rand() % 2;
					fprintf(setting->weight_ptr, "%d ", random);
					weight[weight_c][ifmaps_c][weight_h][weight_w] = random;
#ifdef DEBUG
					printf("%d", weight[weight_c][ifmaps_c][weight_h][weight_w]);
#endif // DEBUG

				}
				fprintf(setting->weight_ptr, "\n");
#ifdef DEBUG
				printf("\n");
#endif // DEBUG
			}
			fprintf(setting->weight_ptr, "\n");
#ifdef DEBUG
			printf("\n");
#endif // DEBUG
		}
		fprintf(setting->weight_ptr, "\n");
#ifdef DEBUG
		printf("\n");
#endif // DEBUG
	}


	for (int bias_cnt = 0; bias_cnt < setting->weight_num; bias_cnt++)
	{
		if (bias_en == 1)
		{
			random = rand() % ((setting->weight_hight * setting->weight_width * setting->ifmaps_channel / 2)) - ((setting->weight_hight * setting->weight_width * setting->ifmaps_channel / 4));
		}
		else
		{
			random = 0;
		}
		fprintf(setting->bias_ptr, "%x ", (random & 0x0000ffff));
		bias[bias_cnt] = random;
#ifdef DEBUG
		printf("%d\n", bias[bias_cnt]);
#endif // DEBUG
	}

	fclose(setting->ifmaps_ptr);
	fclose(setting->weight_ptr);
	fclose(setting->bias_ptr);
	/////////////////////////////////compute////////////////////////////////////

	for (int o_ch = 0; o_ch < setting->weight_num; o_ch++)
	{
		for (int o_h = 0; o_h < ofmaps_hight; o_h++)
		{
			for (int o_w = 0; o_w < ofmaps_width; o_w++)
			{
				int psum = 0;
				for (int w_h = 0; w_h < setting->weight_hight; w_h++)
				{
					for (int w_w = 0; w_w < setting->weight_width; w_w++)
					{
						int h = o_h * setting->stride + w_h;
						int w = o_w * setting->stride + w_w;
						if (h >= 0 && w >= 0 && h < setting->ifmaps_hight && w < setting->ifmaps_width)
						{
							for (int ch = 0; ch < setting->ifmaps_channel; ch++)
							{
#ifdef BITCOUNT
								int tmp = (!(bool)(ifmaps[ch][h][w] ^ weight[o_ch][ch][w_h][w_w])) ? 1 : 0;
#else // bitcount
								int tmp = (!(bool)(ifmaps[ch][h][w] ^ weight[o_ch][ch][w_h][w_w])) ? 1 : -1;
#endif									
								psum += tmp;
							}
						}
					}
				}
				psum_before_bias[o_ch][o_h][o_w] = psum;
				psum_after_bias[o_ch][o_h][o_w] = psum + bias[o_ch];
				fprintf(setting->psum_before_bias_ptr, "%d ", psum);
				fprintf(setting->psum_after_bias_ptr, "%d ", psum + bias[o_ch]);

#ifdef BITCOUNT
				psum = psum >= (setting->weight_hight * setting->weight_width * setting->ifmaps_channel / 2) ? 1 : 0;
#else // bitcount
				psum = ((psum + bias[o_ch]) >= 0) ? 1 : 0;
#endif
				ofmaps[o_ch][o_h][o_w] = psum;
				fprintf(setting->ofmaps_ptr, "%d ", psum);

			}
			fprintf(setting->ofmaps_ptr, "\n");
			fprintf(setting->psum_before_bias_ptr, "\n");
			fprintf(setting->psum_after_bias_ptr, "\n");
		}
		fprintf(setting->ofmaps_ptr, "\n");
		fprintf(setting->psum_before_bias_ptr, "\n");
		fprintf(setting->psum_after_bias_ptr, "\n");
#ifdef PROGRESS
		printf("進度完成第 %d 個weight計算(%d/%d)\n", (o_ch + 1), (o_ch + 1), setting->weight_num);
#endif

	}
	fclose(setting->ofmaps_ptr);
	fclose(setting->psum_before_bias_ptr);
	fclose(setting->psum_after_bias_ptr);

}

void write_pool(data_setting* setting)
{
	bool*** ifmaps;
	bool*** ofmaps;

	int random = 0;
	int ofmaps_width = ((setting->ifmaps_width - setting->weight_width) / setting->stride + 1);
	int ofmaps_hight = ((setting->ifmaps_hight - setting->weight_hight) / setting->stride + 1);

	ifmaps = construct_bool3Darray(setting->ifmaps_channel, setting->ifmaps_hight, setting->ifmaps_width);
	ofmaps = construct_bool3Darray(setting->ifmaps_channel, ofmaps_hight, ofmaps_width);

	srand(time(NULL));

	/////////////////////////////////build data////////////////////////////////////

	for (int ifmaps_c = 0; ifmaps_c < setting->ifmaps_channel; ifmaps_c++)
	{
		for (int ifmaps_h = 0; ifmaps_h < setting->ifmaps_hight; ifmaps_h++)
		{
			for (int ifmaps_w = 0; ifmaps_w < setting->ifmaps_width; ifmaps_w++)
			{
				random = rand() % 2;
				fprintf(setting->ifmaps_ptr, "%d ", random);
				ifmaps[ifmaps_c][ifmaps_h][ifmaps_w] = random;
#ifdef DEBUG
				printf("%d", ifmaps[ifmaps_c][ifmaps_h][ifmaps_w]);
#endif // DEBUG
			}
			fprintf(setting->ifmaps_ptr, "\n");
#ifdef DEBUG
			printf("\n");
#endif // DEBUG
		}
		fprintf(setting->ifmaps_ptr, "\n");
#ifdef DEBUG
		printf("\n");
#endif // DEBUG
	}

	fclose(setting->ifmaps_ptr);
	fclose(setting->weight_ptr);
	fclose(setting->bias_ptr);
	/////////////////////////////////compute////////////////////////////////////

	for (int o_ch = 0; o_ch < setting->ifmaps_channel; o_ch++)
	{
		for (int o_h = 0; o_h < ofmaps_hight; o_h++)
		{
			for (int o_w = 0; o_w < ofmaps_width; o_w++)
			{
				int flag = 0;
				for (int w_h = 0; w_h < setting->weight_hight; w_h++)
				{
					for (int w_w = 0; w_w < setting->weight_width; w_w++)
					{
						int h = o_h * setting->stride + w_h;
						int w = o_w * setting->stride + w_w;
						if (h >= 0 && w >= 0 && h < setting->ifmaps_hight && w < setting->ifmaps_width)
						{
							if (ifmaps[o_ch][h][w] == 1)
							{
								flag = 1;
								break; 
							}
						}
					}
					if (flag == 1)
						break;
				}

				ofmaps[o_ch][o_h][o_w] = flag;
				fprintf(setting->ofmaps_ptr, "%d ", flag);

			}
			fprintf(setting->ofmaps_ptr, "\n");
		}
		fprintf(setting->ofmaps_ptr, "\n");
	}

}

data_setting check_file(int wr_ofmaps)
{
	char buf[50];
	int buf_len = 50;

	data_setting setting;
	
	if ((setting.weight_ptr = fopen("../weight.txt", "rb")) == NULL)
	{
		printf("weight.txt開啟失敗");
		return;
	}
	if ((setting.bias_ptr = fopen("../bias.txt", "rb")) == NULL)
	{
		printf("bias.txt開啟失敗");
		return;
	}
	if ((setting.ifmaps_ptr = fopen("../ifmaps.txt", "rb")) == NULL)
	{
		printf("ifmaps.txt開啟失敗");
		return;
	}
	if (wr_ofmaps == 0)
	{
		if ((setting.psum_before_bias_ptr = fopen("../psum_before_bias.txt", "rb")) == NULL)
		{
			printf("psum_before_bias.txt開啟失敗");
			return;
		}
		if ((setting.psum_after_bias_ptr = fopen("../psum_after_bias.txt", "rb")) == NULL)
		{
			printf("psum_after_bias.txt開啟失敗");
			return;
		}
		if ((setting.ofmaps_ptr = fopen("../ofmaps.txt", "rb")) == NULL)
		{
			printf("ofmaps.txt開啟失敗");
			return;
		}
	}
	else
	{
		if ((setting.psum_before_bias_ptr = fopen("../psum_before_bias.txt", "wb+")) == NULL)
		{
			printf("psum_before_bias.txt開啟失敗");
			return;
		}
		if ((setting.psum_after_bias_ptr = fopen("../psum_after_bias.txt", "wb+")) == NULL)
		{
			printf("psum_after_bias.txt開啟失敗");
			return;
		}
		if ((setting.ofmaps_ptr = fopen("../ofmaps.txt", "wb+")) == NULL)
		{
			printf("ofmaps.txt開啟失敗");
			return;
		}
	}
	
	if ((setting.setting_ptr = fopen("../setting.txt", "rb")) == NULL)
	{
		printf("setting.txt開啟失敗");
		return;
	}

	int cnt = 0;
	while ((fgets(buf, buf_len, setting.setting_ptr)) != NULL) {
		if (buf[0] == '/') continue;
		/////allocate values////
		switch (cnt)
		{
			case 0:
				sscanf_s(buf, "%d", &setting.function);
				if (setting.function == 0)
				{
					printf("這是執行convolution的結果\n");
				}
				else if (setting.function == 1)
				{
					printf("這是執行pooling的結果\n");
				}
				else
				{
					printf("資料有誤\n");
					return;
				}
			break;
			case 1:
				sscanf_s(buf, "%d", &setting.ifmaps_width);
				printf("ifmaps_width = %d\n", setting.ifmaps_width);
			break;
			case 2:
				sscanf_s(buf, "%d", &setting.ifmaps_hight);
				printf("ifmaps_hight = %d\n", setting.ifmaps_hight);
			break;
			case 3:
				sscanf_s(buf, "%d", &setting.ifmaps_channel);
				printf("ifmaps_channel = %d\n", setting.ifmaps_channel);
			break;
			case 4:
				sscanf_s(buf, "%d", &setting.weight_width);
				printf("weight_width = %d\n", setting.weight_width);
				break;
			case 5:
				sscanf_s(buf, "%d", &setting.weight_hight);
				printf("weight_hight = %d\n", setting.weight_hight);
				break;
			case 6:
				sscanf_s(buf, "%d", &setting.weight_num);
				printf("weight_num = %d\n", setting.weight_num);
				break;
			case 7:
				sscanf_s(buf, "%d", &setting.stride);
				printf("stride = %d\n", setting.stride);
				break;
		default:
			break;
		}
		cnt++;
	}
	printf("資料無誤\n");
	return setting;
}

bool*** construct_bool3Darray(int z, int y, int x)
{
	bool*** ptr;
	ptr = (bool***)malloc(sizeof(bool**) * z);
	for (int q = 0; q < z; q++)
	{
		ptr[q] = (bool**)malloc(sizeof(bool*) * y);
		for (int w = 0; w < y; w++)
		{
			ptr[q][w] = (bool*)malloc(sizeof(bool) * x);
		}
	}
	return ptr;
}

bool**** construct_bool4Darray(int w, int z, int y, int x)
{
	bool**** ptr;
	ptr = (bool****)malloc(sizeof(bool***) * w);
	for (int q = 0; q < w; q++)
	{
		ptr[q] = (bool***)malloc(sizeof(bool**) * z);
		for (int w = 0; w < z; w++)
		{
			ptr[q][w] = (bool**)malloc(sizeof(bool*) * y);
			for (int e = 0; e < x; e++)
			{
				ptr[q][w][e] = (bool*)malloc(sizeof(bool) * x);
			}
		}
	}
	return ptr;
}

int*** construct_int3Darray(int z, int y, int x)
{
	int*** ptr;
	ptr = (int***)malloc(sizeof(int**) * z);
	for (int q = 0; q < z; q++)
	{
		ptr[q] = (int**)malloc(sizeof(int*) * y);
		for (int w = 0; w < y; w++)
		{
			ptr[q][w] = (int*)malloc(sizeof(int) * x);
		}
	}
	return ptr;
}

int* construct_int1Darray(int x)
{
	int* ptr;
	ptr = (int*)malloc(sizeof(int) * x);
	return ptr;
}