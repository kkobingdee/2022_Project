/*                                *\
      Non_KeySchedule_Masking
\*                                */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "aes.h"
#include <time.h>

#define MUL2(a) (a<<1)^(a&0x80?0x1b:0)
#define MUL3(a) (MUL2(a))^(a)
#define MUL4(a) MUL2((MUL2(a)))
#define MUL8(a) MUL2((MUL2((MUL2(a)))))
#define MUL9(a) (MUL8(a))^(a)
#define MULB(a) (MUL8(a))^(MUL2(a))^(a)
#define MULD(a) (MUL8(a))^(MUL4(a))^(a)
#define MULE(a) (MUL8(a))^(MUL4(a))^(MUL2(a))

#define RotWord(x) ((x << 8) | (x >> 24))
#define SubWord(x)								\
	((u32)Sbox[(u8)(x >> 24)] << 24)			\
	| ((u32)Sbox[(u8)(x >> 16) & 0xff] << 16)	\
	| ((u32)Sbox[(u8)(x >> 8) & 0xff] << 8)		\
	| ((u32)Sbox[(u8)(x & 0xff)])				\

static u8 M[10] = { 0x00, };
static u8 Mbox[256] = { 0x00, };

void AddRoundKey(u8 S[16], u8 RK[16]) {
	S[0] ^= RK[0]; S[1] ^= RK[1]; S[2] ^= RK[2]; S[3] ^= RK[3];
	S[4] ^= RK[4]; S[5] ^= RK[5]; S[6] ^= RK[6]; S[7] ^= RK[7];
	S[8] ^= RK[8]; S[9] ^= RK[9]; S[10] ^= RK[10]; S[11] ^= RK[11];
	S[12] ^= RK[12]; S[13] ^= RK[13]; S[14] ^= RK[14]; S[15] ^= RK[15];
}

void SubBytes(u8 S[16]) {
	S[0] = Sbox[S[0]]; S[1] = Sbox[S[1]]; S[2] = Sbox[S[2]]; S[3] = Sbox[S[3]];
	S[4] = Sbox[S[4]]; S[5] = Sbox[S[5]]; S[6] = Sbox[S[6]]; S[7] = Sbox[S[7]];
	S[8] = Sbox[S[8]]; S[9] = Sbox[S[9]]; S[10] = Sbox[S[10]]; S[11] = Sbox[S[11]];
	S[12] = Sbox[S[12]]; S[13] = Sbox[S[13]]; S[14] = Sbox[S[14]]; S[15] = Sbox[S[15]];
}

void ShiftRows(u8 S[16]) {
	u8 temp;
	temp = S[1]; S[1] = S[5]; S[5] = S[9]; S[9] = S[13]; S[13] = temp;
	temp = S[2]; S[2] = S[10]; S[10] = temp; temp = S[6]; S[6] = S[14]; S[14] = temp;
	temp = S[15]; S[15] = S[11]; S[11] = S[7]; S[7] = S[3]; S[3] = temp;
}

void MixColums(u8 S[16]) {
	u8 temp[16];
	int i;

	for (i = 0; i < 16; i += 4) {
		temp[i] = MUL2(S[i]) ^ MUL3(S[i + 1]) ^ S[i + 2] ^ S[i + 3];
		temp[i + 1] = S[i] ^ MUL2(S[i + 1]) ^ MUL3(S[i + 2]) ^ S[i + 3];
		temp[i + 2] = S[i] ^ S[i + 1] ^ MUL2(S[i + 2]) ^ MUL3(S[i + 3]);
		temp[i + 3] = MUL3(S[i]) ^ S[i + 1] ^ S[i + 2] ^ MUL2(S[i + 3]);
	}

	S[0] = temp[0]; S[1] = temp[1]; S[2] = temp[2]; S[3] = temp[3];
	S[4] = temp[4]; S[5] = temp[5]; S[6] = temp[6]; S[7] = temp[7];
	S[8] = temp[8]; S[9] = temp[9]; S[10] = temp[10]; S[11] = temp[11];
	S[12] = temp[12]; S[13] = temp[13]; S[14] = temp[14]; S[15] = temp[15];
}

void Mixcolums_M() {
	M[6] = MUL2(M[2]) ^ MUL3(M[3]) ^ M[4] ^ M[5];
	M[7] = M[2] ^ MUL2(M[3]) ^ MUL3(M[4]) ^ M[5];
	M[8] = M[2] ^ M[3] ^ MUL2(M[4]) ^ MUL3(M[5]);
	M[9] = MUL3(M[2]) ^ M[3] ^ M[4] ^ MUL2(M[5]);
}

void getM() {
	srand(time(NULL));
	for (int i = 0; i < 6; i++) {
		M[i] = rand() % 0xff;
	}
	Mixcolums_M();
}

void getMbox() {
	for (int i = 0; i < 256; i++) {
		Mbox[i ^ M[0]] = Sbox[i] ^ M[1];
	}
}

void AddRoundKey_M(u8 S[16], u8 RK[16]) {
	S[0] ^= (RK[0] ^ M[6] ^ M[0]); S[1] ^= (RK[1] ^ M[7] ^ M[0]); S[2] ^= (RK[2] ^ M[8] ^ M[0]); S[3] ^= (RK[3] ^ M[9] ^ M[0]);
	S[4] ^= (RK[4] ^ M[6] ^ M[0]); S[5] ^= (RK[5] ^ M[7] ^ M[0]); S[6] ^= (RK[6] ^ M[8] ^ M[0]); S[7] ^= (RK[7] ^ M[9] ^ M[0]);
	S[8] ^= (RK[8] ^ M[6] ^ M[0]); S[9] ^= (RK[9] ^ M[7] ^ M[0]); S[10] ^= (RK[10] ^ M[8] ^ M[0]); S[11] ^= (RK[11] ^ M[9] ^ M[0]);
	S[12] ^= (RK[12] ^ M[6] ^ M[0]); S[13] ^= (RK[13] ^ M[7] ^ M[0]); S[14] ^= (RK[14] ^ M[8] ^ M[0]); S[15] ^= (RK[15] ^ M[9] ^ M[0]);
}

void AddRoundKey_M2(u8 S[16], u8 RK[16]) {
	S[0] ^= RK[0] ^ M[1]; S[1] ^= RK[1] ^ M[1]; S[2] ^= RK[2] ^ M[1]; S[3] ^= RK[3] ^ M[1];
	S[4] ^= RK[4] ^ M[1]; S[5] ^= RK[5] ^ M[1]; S[6] ^= RK[6] ^ M[1]; S[7] ^= RK[7] ^ M[1];
	S[8] ^= RK[8] ^ M[1]; S[9] ^= RK[9] ^ M[1]; S[10] ^= RK[10] ^ M[1]; S[11] ^= RK[11] ^ M[1];
	S[12] ^= RK[12] ^ M[1]; S[13] ^= RK[13] ^ M[1]; S[14] ^= RK[14] ^ M[1]; S[15] ^= RK[15] ^ M[1];
}

void SubBytes_M(u8 S[16]) {
	S[0] = Mbox[S[0]]; S[1] = Mbox[S[1]]; S[2] = Mbox[S[2]]; S[3] = Mbox[S[3]];
	S[4] = Mbox[S[4]]; S[5] = Mbox[S[5]]; S[6] = Mbox[S[6]]; S[7] = Mbox[S[7]];
	S[8] = Mbox[S[8]]; S[9] = Mbox[S[9]]; S[10] = Mbox[S[10]]; S[11] = Mbox[S[11]];
	S[12] = Mbox[S[12]]; S[13] = Mbox[S[13]]; S[14] = Mbox[S[14]]; S[15] = Mbox[S[15]];
}

void Pre_MixC(u8 S[16]) {
	for (int i = 0; i < 16; i += 4) {
		S[i] ^= M[1] ^ M[2];
		S[i + 1] ^= M[1] ^ M[3];
		S[i + 2] ^= M[1] ^ M[4];
		S[i + 3] ^= M[1] ^ M[5];
	}
}

void AES_MASK_ENC(u8 PT[16], u8 RK[], u8 CT[16], int keysize) {
	int Nr = keysize / 32 + 6;
	int i;
	u8 temp[16];

	getM();
	getMbox();

	for (i = 0; i < 16; i += 4) {
		temp[i] = PT[i] ^ M[6];
		temp[i + 1] = PT[i + 1] ^ M[7];
		temp[i + 2] = PT[i + 2] ^ M[8];
		temp[i + 3] = PT[i + 3] ^ M[9];
	}

	AddRoundKey_M(temp, RK);

	for (i = 0; i < Nr - 1; i++) {
		SubBytes_M(temp);						//M0 <-> M1
		ShiftRows(temp);
		Pre_MixC(temp);							//M1 <-> M2, M3, M4, M5
		MixColums(temp);						//M2, M3, M4, M5 <-> M6, M7, M8, M9
		AddRoundKey_M(temp, RK + 16 * (i + 1));	//M6, M7, M8, M9 <-> M0
	}

	SubBytes_M(temp);
	ShiftRows(temp);
	AddRoundKey_M2(temp, RK + 16 * (i + 1));	//M1 delete !!

	for (i = 0; i < 16; i++) {
		CT[i] = temp[i];
	}
}

u32 u4byte_in(u8* x) {
	return (x[0] << 24) | (x[1] << 16) | (x[2] << 8) | x[3];
}

void u4byte_out(u8* x, u32 y) {
	x[0] = (y >> 24) & 0xff;
	x[1] = (y >> 16) & 0xff;
	x[2] = (y >> 8) & 0xff;
	x[3] = y & 0xff;
}

void AES_KeyWordToByte(u32 W[], u8 RK[]) {
	int i;
	for (i = 0; i < 44; i++) {
		u4byte_out(RK + 4 * i, W[i]);
	}
}

u32 Rcons[10] = { 0x01000000, 0x02000000, 0x04000000, 0x08000000, 0x10000000, 0x20000000, 0x40000000, 0x80000000, 0x1b000000, 0x36000000 };

void RoundKeyGeneration128(u8 MK[], u8 RK[]) {
	u32 W[44];
	int i;
	u32 T;

	W[0] = u4byte_in(MK);
	W[1] = u4byte_in(MK + 4);
	W[2] = u4byte_in(MK + 8);
	W[3] = u4byte_in(MK + 12);

	for (i = 0; i < 10; i++) {
		//T = G_func(W[4 * i + 3]);
		T = W[4 * i + 3];
		T = RotWord(T);
		T = SubWord(T);
		T ^= Rcons[i];

		W[4 * i + 4] = W[4 * i] ^ T;
		W[4 * i + 5] = W[4 * i + 1] ^ W[4 * i + 4];
		W[4 * i + 6] = W[4 * i + 2] ^ W[4 * i + 5];
		W[4 * i + 7] = W[4 * i + 3] ^ W[4 * i + 6];
	}
	AES_KeyWordToByte(W, RK);
}

void AES_KeySchedule(u8 MK[], u8 RK[], int keysize) {
	if (keysize == 128) {
		RoundKeyGeneration128(MK, RK);
	}
	/*if (keysize == 192) {
		RoundKeyGeneration192(MK, RK);
	}
	if (keysize == 256) {
		RoundKeyGeneration256(MK, RK);
	}*/
}


int main() {
	u8 PT[16] = { 0x7F, 0x14, 0x58, 0x23, 0xDB, 0xE1, 0xD6, 0xDF, 0xE5, 0xCA, 0x92, 0xDF, 0x8C, 0x92, 0xDA, 0x85 };//평문
	u8 MK[16] = { 0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c }; //마스터키
	u8 CT[16] = { 0x00, };
	u8 RK[240] = { 0x00, };
	int keysize = 128;

	

	AES_KeySchedule(MK, RK, keysize);
	AES_MASK_ENC(PT, RK, CT, keysize);
	for (int i = 0; i < 16; i++) {
		printf("%02x ", CT[i]);
	}
	

	return 0;
}
