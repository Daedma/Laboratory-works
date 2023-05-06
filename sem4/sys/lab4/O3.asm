	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"main.cc"
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.seh_proc main
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	pushq	%rdi
	.seh_pushreg %rdi
	subq	$104, %rsp
	.seh_stackalloc 104
	movdqa	%xmm11, 80(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm11, 80
	movdqa	%xmm10, 64(%rsp)                # 16-byte Spill
	.seh_savexmm %xmm10, 64
	movdqa	%xmm9, 48(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm9, 48
	movdqa	%xmm8, 32(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm8, 32
	movdqa	%xmm7, 16(%rsp)                 # 16-byte Spill
	.seh_savexmm %xmm7, 16
	movdqa	%xmm6, (%rsp)                   # 16-byte Spill
	.seh_savexmm %xmm6, 0
	.seh_endprologue
	movl	"?size@@3HA"(%rip), %r9d
	testq	%r9, %r9
	je	.LBB0_17
# %bb.1:
	movq	"?a@@3PEAHEA"(%rip), %r11
	movl	"?res@@3HA"(%rip), %eax
	cmpl	$8, %r9d
	jb	.LBB0_2
# %bb.3:
	leaq	(%r11,%r9,4), %rcx
	leaq	"?res@@3HA"(%rip), %rdx
	cmpq	%rdx, %rcx
	jbe	.LBB0_6
# %bb.4:
	leaq	"?res@@3HA"+4(%rip), %rcx
	cmpq	%rcx, %r11
	jae	.LBB0_6
.LBB0_2:
	xorl	%edi, %edi
.LBB0_13:
	movq	%rdi, %r8
	notq	%r8
	addq	%r9, %r8
	movq	%r9, %rcx
	andq	$3, %rcx
	je	.LBB0_15
	.p2align	4, 0x90
.LBB0_14:                               # =>This Inner Loop Header: Depth=1
	movl	(%r11,%rdi,4), %esi
	movl	%esi, %edx
	imull	%esi, %edx
	imull	%esi, %edx
	addl	%edx, %eax
	movl	%eax, "?res@@3HA"(%rip)
	incq	%rdi
	decq	%rcx
	jne	.LBB0_14
.LBB0_15:
	cmpq	$3, %r8
	jb	.LBB0_17
	.p2align	4, 0x90
.LBB0_16:                               # =>This Inner Loop Header: Depth=1
	movl	(%r11,%rdi,4), %ecx
	movl	%ecx, %edx
	imull	%ecx, %edx
	imull	%ecx, %edx
	addl	%eax, %edx
	movl	%edx, "?res@@3HA"(%rip)
	movl	4(%r11,%rdi,4), %eax
	movl	%eax, %ecx
	imull	%eax, %ecx
	imull	%eax, %ecx
	addl	%edx, %ecx
	movl	%ecx, "?res@@3HA"(%rip)
	movl	8(%r11,%rdi,4), %eax
	movl	%eax, %edx
	imull	%eax, %edx
	imull	%eax, %edx
	addl	%ecx, %edx
	movl	%edx, "?res@@3HA"(%rip)
	movl	12(%r11,%rdi,4), %ecx
	movl	%ecx, %eax
	imull	%ecx, %eax
	imull	%ecx, %eax
	addl	%edx, %eax
	movl	%eax, "?res@@3HA"(%rip)
	addq	$4, %rdi
	cmpq	%rdi, %r9
	jne	.LBB0_16
	jmp	.LBB0_17
.LBB0_6:
	movl	%r9d, %edi
	andl	$-8, %edi
	movd	%eax, %xmm0
	leaq	-8(%rdi), %rax
	movq	%rax, %r8
	shrq	$3, %r8
	incq	%r8
	testq	%rax, %rax
	je	.LBB0_18
# %bb.7:
	movq	%r8, %r10
	andq	$-2, %r10
	pxor	%xmm1, %xmm1
	xorl	%eax, %eax
	.p2align	4, 0x90
.LBB0_8:                                # =>This Inner Loop Header: Depth=1
	movdqu	(%r11,%rax,4), %xmm2
	movdqu	16(%r11,%rax,4), %xmm6
	movdqu	32(%r11,%rax,4), %xmm9
	movdqu	48(%r11,%rax,4), %xmm8
	pshufd	$245, %xmm2, %xmm11             # xmm11 = xmm2[1,1,3,3]
	movdqa	%xmm11, %xmm7
	pmuludq	%xmm11, %xmm7
	movdqa	%xmm2, %xmm4
	pmuludq	%xmm2, %xmm4
	pshufd	$245, %xmm6, %xmm10             # xmm10 = xmm6[1,1,3,3]
	movdqa	%xmm10, %xmm3
	pmuludq	%xmm10, %xmm3
	movdqa	%xmm6, %xmm5
	pmuludq	%xmm6, %xmm5
	pmuludq	%xmm2, %xmm4
	pshufd	$232, %xmm4, %xmm4              # xmm4 = xmm4[0,2,2,3]
	pmuludq	%xmm11, %xmm7
	pshufd	$232, %xmm7, %xmm2              # xmm2 = xmm7[0,2,2,3]
	punpckldq	%xmm2, %xmm4            # xmm4 = xmm4[0],xmm2[0],xmm4[1],xmm2[1]
	paddd	%xmm0, %xmm4
	pmuludq	%xmm6, %xmm5
	pshufd	$232, %xmm5, %xmm5              # xmm5 = xmm5[0,2,2,3]
	pmuludq	%xmm10, %xmm3
	pshufd	$232, %xmm3, %xmm0              # xmm0 = xmm3[0,2,2,3]
	punpckldq	%xmm0, %xmm5            # xmm5 = xmm5[0],xmm0[0],xmm5[1],xmm0[1]
	paddd	%xmm1, %xmm5
	pshufd	$245, %xmm9, %xmm1              # xmm1 = xmm9[1,1,3,3]
	movdqa	%xmm1, %xmm2
	pmuludq	%xmm1, %xmm2
	movdqa	%xmm9, %xmm0
	pmuludq	%xmm9, %xmm0
	pshufd	$245, %xmm8, %xmm3              # xmm3 = xmm8[1,1,3,3]
	movdqa	%xmm3, %xmm6
	pmuludq	%xmm3, %xmm6
	movdqa	%xmm8, %xmm7
	pmuludq	%xmm8, %xmm7
	pmuludq	%xmm9, %xmm0
	pshufd	$232, %xmm0, %xmm0              # xmm0 = xmm0[0,2,2,3]
	pmuludq	%xmm1, %xmm2
	pshufd	$232, %xmm2, %xmm1              # xmm1 = xmm2[0,2,2,3]
	punpckldq	%xmm1, %xmm0            # xmm0 = xmm0[0],xmm1[0],xmm0[1],xmm1[1]
	paddd	%xmm4, %xmm0
	pmuludq	%xmm8, %xmm7
	pshufd	$232, %xmm7, %xmm1              # xmm1 = xmm7[0,2,2,3]
	pmuludq	%xmm3, %xmm6
	pshufd	$232, %xmm6, %xmm2              # xmm2 = xmm6[0,2,2,3]
	punpckldq	%xmm2, %xmm1            # xmm1 = xmm1[0],xmm2[0],xmm1[1],xmm2[1]
	paddd	%xmm5, %xmm1
	addq	$16, %rax
	addq	$-2, %r10
	jne	.LBB0_8
# %bb.9:
	testb	$1, %r8b
	je	.LBB0_11
.LBB0_10:
	movdqu	(%r11,%rax,4), %xmm2
	movdqu	16(%r11,%rax,4), %xmm4
	pshufd	$245, %xmm2, %xmm9              # xmm9 = xmm2[1,1,3,3]
	movdqa	%xmm9, %xmm5
	pmuludq	%xmm9, %xmm5
	movdqa	%xmm2, %xmm6
	pmuludq	%xmm2, %xmm6
	pshufd	$245, %xmm4, %xmm8              # xmm8 = xmm4[1,1,3,3]
	movdqa	%xmm8, %xmm7
	pmuludq	%xmm8, %xmm7
	movdqa	%xmm4, %xmm3
	pmuludq	%xmm4, %xmm3
	pmuludq	%xmm2, %xmm6
	pshufd	$232, %xmm6, %xmm2              # xmm2 = xmm6[0,2,2,3]
	pmuludq	%xmm9, %xmm5
	pshufd	$232, %xmm5, %xmm5              # xmm5 = xmm5[0,2,2,3]
	punpckldq	%xmm5, %xmm2            # xmm2 = xmm2[0],xmm5[0],xmm2[1],xmm5[1]
	paddd	%xmm2, %xmm0
	pmuludq	%xmm4, %xmm3
	pshufd	$232, %xmm3, %xmm2              # xmm2 = xmm3[0,2,2,3]
	pmuludq	%xmm8, %xmm7
	pshufd	$232, %xmm7, %xmm3              # xmm3 = xmm7[0,2,2,3]
	punpckldq	%xmm3, %xmm2            # xmm2 = xmm2[0],xmm3[0],xmm2[1],xmm3[1]
	paddd	%xmm2, %xmm1
.LBB0_11:
	paddd	%xmm1, %xmm0
	pshufd	$238, %xmm0, %xmm1              # xmm1 = xmm0[2,3,2,3]
	paddd	%xmm0, %xmm1
	pshufd	$85, %xmm1, %xmm0               # xmm0 = xmm1[1,1,1,1]
	paddd	%xmm1, %xmm0
	movd	%xmm0, "?res@@3HA"(%rip)
	cmpq	%r9, %rdi
	jne	.LBB0_12
.LBB0_17:
	xorl	%eax, %eax
	movaps	(%rsp), %xmm6                   # 16-byte Reload
	movaps	16(%rsp), %xmm7                 # 16-byte Reload
	movaps	32(%rsp), %xmm8                 # 16-byte Reload
	movaps	48(%rsp), %xmm9                 # 16-byte Reload
	movaps	64(%rsp), %xmm10                # 16-byte Reload
	movaps	80(%rsp), %xmm11                # 16-byte Reload
	addq	$104, %rsp
	popq	%rdi
	popq	%rsi
	retq
.LBB0_12:
	movd	%xmm0, %eax
	jmp	.LBB0_13
.LBB0_18:
	pxor	%xmm1, %xmm1
	xorl	%eax, %eax
	testb	$1, %r8b
	jne	.LBB0_10
	jmp	.LBB0_11
	.seh_endproc
                                        # -- End function
	.bss
	.globl	"?res@@3HA"                     # @"?res@@3HA"
	.p2align	2
"?res@@3HA":
	.long	0                               # 0x0

	.globl	"?d@@3HA"                       # @"?d@@3HA"
	.p2align	2
"?d@@3HA":
	.long	0                               # 0x0

	.globl	"?size@@3HA"                    # @"?size@@3HA"
	.p2align	2
"?size@@3HA":
	.long	0                               # 0x0

	.globl	"?a@@3PEAHEA"                   # @"?a@@3PEAHEA"
	.p2align	3
"?a@@3PEAHEA":
	.quad	0

	.addrsig
