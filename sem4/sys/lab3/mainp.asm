	.text
	.def	@feat.00;
	.scl	3;
	.type	0;
	.endef
	.globl	@feat.00
.set @feat.00, 0
	.file	"mainp.cpp"
	.def	"?f@@YAXHH@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	"?f@@YAXHH@Z"                   # -- Begin function ?f@@YAXHH@Z
	.p2align	4, 0x90
"?f@@YAXHH@Z":                          # @"?f@@YAXHH@Z"
.seh_proc "?f@@YAXHH@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movl	%edx, 4(%rsp)
	movl	%ecx, (%rsp)
	movl	(%rsp), %edx
	movl	4(%rsp), %r9d
	#APP
	movl	%edx, %eax
	movl	%r9d, %r8d
	movl	%eax, %ecx
	subl	%r8d, %ecx
	jle	less_equal
	addl	%eax, %r8d
	movl	%ecx, %eax
	jmp	division
less_equal:
	jne	less
	imull	%eax, %eax
	negl	%eax
	jmp	end
less:
	leal	(%rax,%rax,2), %eax
	addl	$-2, %eax
division:
	cltd
	idivl	%r8d
end:
	#NO_APP
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?f_cpp@@YAHHH@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	"?f_cpp@@YAHHH@Z"               # -- Begin function ?f_cpp@@YAHHH@Z
	.p2align	4, 0x90
"?f_cpp@@YAHHH@Z":                      # @"?f_cpp@@YAHHH@Z"
.seh_proc "?f_cpp@@YAHHH@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movl	%edx, 8(%rsp)
	movl	%ecx, 4(%rsp)
	movl	4(%rsp), %eax
	cmpl	8(%rsp), %eax
	jle	.LBB1_2
# %bb.1:
	movl	4(%rsp), %eax
	subl	8(%rsp), %eax
	movl	4(%rsp), %ecx
	addl	8(%rsp), %ecx
	cltd
	idivl	%ecx
	movl	%eax, 12(%rsp)
	jmp	.LBB1_9
.LBB1_2:
	movl	4(%rsp), %eax
	cmpl	8(%rsp), %eax
	jne	.LBB1_4
# %bb.3:
	xorl	%eax, %eax
	subl	4(%rsp), %eax
	imull	8(%rsp), %eax
	movl	%eax, 12(%rsp)
	jmp	.LBB1_9
.LBB1_4:
	movl	4(%rsp), %eax
	cmpl	8(%rsp), %eax
	jge	.LBB1_6
# %bb.5:
	imull	$3, 4(%rsp), %eax
	subl	$2, %eax
	cltd
	idivl	8(%rsp)
	movl	%eax, 12(%rsp)
	jmp	.LBB1_9
.LBB1_6:
	jmp	.LBB1_7
.LBB1_7:
	jmp	.LBB1_8
.LBB1_8:
	movl	$0, 12(%rsp)
.LBB1_9:
	movl	12(%rsp), %eax
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	main;
	.scl	2;
	.type	32;
	.endef
	.globl	main                            # -- Begin function main
	.p2align	4, 0x90
main:                                   # @main
.seh_proc main
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	leaq	"?cin@std@@3V?$basic_istream@DU?$char_traits@D@std@@@1@A"(%rip), %rcx
	leaq	52(%rsp), %rdx
	callq	"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z"
	movq	%rax, %rcx
	leaq	48(%rsp), %rdx
	callq	"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z"
	movl	48(%rsp), %edx
	movl	52(%rsp), %ecx
	callq	"?f@@YAXHH@Z"
	#APP
	movl	%eax, %eax

	#NO_APP
	movl	%eax, 44(%rsp)
	movl	44(%rsp), %edx
	leaq	"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A"(%rip), %rcx
	callq	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	movq	%rax, %rcx
	leaq	"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(%rip), %rdx
	callq	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z"
	movl	48(%rsp), %edx
	movl	52(%rsp), %ecx
	callq	"?f_cpp@@YAHHH@Z"
	movl	%eax, %edx
	leaq	"?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A"(%rip), %rcx
	callq	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	movq	%rax, %rcx
	leaq	"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"(%rip), %rdx
	callq	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z"
	xorl	%eax, %eax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z"
	.globl	"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z" # -- Begin function ??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z
	.p2align	4, 0x90
"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z": # @"??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z"
.seh_proc "??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rax
	movl	(%rax), %eax
	movl	%eax, 52(%rsp)
	leaq	52(%rsp), %rdx
	callq	"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	movl	52(%rsp), %edx
	movq	64(%rsp), %rcx
	movl	%edx, (%rcx)
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.globl	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z" # -- Begin function ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z
	.p2align	4, 0x90
"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z": # @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
.Lfunc_begin0:
.seh_proc "??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$224, %rsp
	.seh_stackalloc 224
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 88(%rbp)
	movl	%edx, 84(%rbp)
	movq	%rcx, 72(%rbp)
	movq	72(%rbp), %rdx
	movq	%rdx, -24(%rbp)                 # 8-byte Spill
	movl	$0, 68(%rbp)
	leaq	48(%rbp), %rcx
	callq	"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	leaq	48(%rbp), %rcx
	callq	"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB4_1
	jmp	.LBB4_15
.LBB4_1:
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	leaq	24(%rbp), %rdx
	movq	%rdx, -40(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
.Ltmp0:
	callq	"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
.Ltmp1:
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	jmp	.LBB4_2
.LBB4_2:
	leaq	24(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, 40(%rbp)
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$3584, %eax                     # imm = 0xE00
	movl	%eax, 20(%rbp)
	cmpl	$1024, 20(%rbp)                 # imm = 0x400
	je	.LBB4_4
# %bb.3:
	cmpl	$2048, 20(%rbp)                 # imm = 0x800
	jne	.LBB4_6
.LBB4_4:
	movl	84(%rbp), %eax
	movl	%eax, 16(%rbp)
	jmp	.LBB4_7
.LBB4_6:
	movl	84(%rbp), %eax
	movl	%eax, 16(%rbp)
.LBB4_7:
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	40(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movl	16(%rbp), %eax
	movl	%eax, -72(%rbp)                 # 4-byte Spill
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movb	%al, -65(%rbp)                  # 1-byte Spill
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, %rdx
	leaq	-16(%rbp), %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	callq	"??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z"
	movl	-72(%rbp), %r10d                # 4-byte Reload
	movb	-65(%rbp), %dl                  # 1-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	-56(%rbp), %r8                  # 8-byte Reload
	movq	-48(%rbp), %r9                  # 8-byte Reload
.Ltmp2:
	movq	%rsp, %rax
	movl	%r10d, 40(%rax)
	movb	%dl, 32(%rax)
	movq	%rbp, %rdx
	callq	"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
.Ltmp3:
	jmp	.LBB4_12
.LBB4_10:                               # Block address taken
$ehgcr_4_10:
	jmp	.LBB4_11
.LBB4_11:
	jmp	.LBB4_15
.LBB4_12:
	movq	%rbp, %rcx
	callq	"?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB4_13
	jmp	.LBB4_14
.LBB4_13:
	movl	68(%rbp), %eax
	orl	$4, %eax
	movl	%eax, 68(%rbp)
.LBB4_14:
	jmp	.LBB4_11
.LBB4_15:
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	68(%rbp), %edx
.Ltmp6:
	xorl	%eax, %eax
	movb	%al, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp7:
	jmp	.LBB4_16
.LBB4_16:
	leaq	48(%rbp), %rcx
	callq	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	movq	-24(%rbp), %rax                 # 8-byte Reload
	addq	$224, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z")@IMGREL
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.seh_endproc
	.def	"?dtor$5@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$5@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA":
.seh_proc "?dtor$5@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"
.LBB4_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	24(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.seh_endproc
	.def	"?catch$8@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?catch$8@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA":
.seh_proc "?catch$8@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"
	.seh_handler __CxxFrameHandler3, @unwind, @except
.LBB4_8:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp4:
	movl	$4, %edx
	movb	$1, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp5:
	jmp	.LBB4_9
.LBB4_9:
	leaq	.LBB4_10(%rip), %rax
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CATCHRET
	.seh_handlerdata
	.long	("$cppxdata$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z")@IMGREL
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.seh_endproc
	.def	"?dtor$17@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$17@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA":
.seh_proc "?dtor$17@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"
.LBB4_17:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	48(%rbp), %rcx
	callq	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end0:
	.seh_handlerdata
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.p2align	2
"$cppxdata$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z":
	.long	429065506                       # MagicNumber
	.long	4                               # MaxState
	.long	("$stateUnwindMap$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z")@IMGREL # UnwindMap
	.long	1                               # NumTryBlocks
	.long	("$tryMap$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z")@IMGREL # TryBlockMap
	.long	6                               # IPMapEntries
	.long	("$ip2state$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z")@IMGREL # IPToStateXData
	.long	216                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z":
	.long	-1                              # ToState
	.long	"?dtor$17@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	0                               # Action
	.long	0                               # ToState
	.long	0                               # Action
	.long	0                               # ToState
	.long	"?dtor$5@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"@IMGREL # Action
"$tryMap$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z":
	.long	1                               # TryLow
	.long	1                               # TryHigh
	.long	2                               # CatchHigh
	.long	1                               # NumCatches
	.long	("$handlerMap$0$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z")@IMGREL # HandlerArray
"$handlerMap$0$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z":
	.long	64                              # Adjectives
	.long	0                               # Type
	.long	0                               # CatchObjOffset
	.long	"?catch$8@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"@IMGREL # Handler
	.long	72                              # ParentFrameOffset
"$ip2state$??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z":
	.long	.Lfunc_begin0@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp0@IMGREL+1                 # IP
	.long	3                               # ToState
	.long	.Ltmp2@IMGREL+1                 # IP
	.long	1                               # ToState
	.long	.Ltmp6@IMGREL+1                 # IP
	.long	0                               # ToState
	.long	.Ltmp7@IMGREL+1                 # IP
	.long	-1                              # ToState
	.long	"?catch$8@?0???6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z@4HA"@IMGREL # IP
	.long	2                               # ToState
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
                                        # -- End function
	.def	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z"
	.globl	"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z" # -- Begin function ??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z
	.p2align	4, 0x90
"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z": # @"??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z"
.seh_proc "??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	callq	*48(%rsp)
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"
	.globl	"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z" # -- Begin function ??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z
	.p2align	4, 0x90
"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z": # @"??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"
.seh_proc "??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	$0, %rax
	addq	%rax, %rcx
	movl	$10, %edx
	callq	"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movb	%al, %dl
	callq	"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	movq	48(%rsp), %rcx
	callq	"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	movq	48(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
	.globl	"??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ" # -- Begin function ??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ
	.p2align	4, 0x90
"??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ": # @"??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
.seh_proc "??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	leaq	"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"(%rip), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??0id@locale@std@@QEAA@_K@Z"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0id@locale@std@@QEAA@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0id@locale@std@@QEAA@_K@Z"
	.globl	"??0id@locale@std@@QEAA@_K@Z"   # -- Begin function ??0id@locale@std@@QEAA@_K@Z
	.p2align	4, 0x90
"??0id@locale@std@@QEAA@_K@Z":          # @"??0id@locale@std@@QEAA@_K@Z"
.seh_proc "??0id@locale@std@@QEAA@_K@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	%rcx, (%rax)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ"
	.globl	"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ" # -- Begin function ??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ
	.p2align	4, 0x90
"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ": # @"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ"
.seh_proc "??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	leaq	"?id@?$numpunct@D@std@@2V0locale@2@A"(%rip), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??0id@locale@std@@QEAA@_K@Z"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
	.globl	"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ" # -- Begin function ??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ
	.p2align	4, 0x90
"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ": # @"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
.seh_proc "??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	leaq	"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"(%rip), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??0id@locale@std@@QEAA@_K@Z"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.globl	"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z" # -- Begin function ??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z
	.p2align	4, 0x90
"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z": # @"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
.Lfunc_begin1:
.seh_proc "??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$224, %rsp
	.seh_stackalloc 224
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 88(%rbp)
	movq	%rdx, 80(%rbp)
	movq	%rcx, 72(%rbp)
	movq	72(%rbp), %rdx
	movq	%rdx, -24(%rbp)                 # 8-byte Spill
	movl	$0, 68(%rbp)
	leaq	48(%rbp), %rcx
	xorl	%r8d, %r8d
	callq	"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
	leaq	48(%rbp), %rcx
	callq	"??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB11_1
	jmp	.LBB11_9
.LBB11_1:
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	leaq	32(%rbp), %rdx
	movq	%rdx, -40(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
.Ltmp8:
	callq	"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
.Ltmp9:
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	jmp	.LBB11_2
.LBB11_2:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	80(%rbp), %rcx
	movq	%rcx, -72(%rbp)                 # 8-byte Spill
	movq	(%rax), %rcx
	movslq	4(%rcx), %rcx
	addq	%rcx, %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	leaq	16(%rbp), %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	callq	"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	movq	-24(%rbp), %rdx                 # 8-byte Reload
	movq	%rbp, %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	callq	"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z"
	movq	-72(%rbp), %r10                 # 8-byte Reload
	movq	-64(%rbp), %rdx                 # 8-byte Reload
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	-56(%rbp), %r8                  # 8-byte Reload
	movq	-48(%rbp), %r9                  # 8-byte Reload
.Ltmp10:
	movq	%rsp, %rax
	movq	%r10, 48(%rax)
	leaq	68(%rbp), %r10
	movq	%r10, 40(%rax)
	movq	%rdx, 32(%rax)
	leaq	-16(%rbp), %rdx
	callq	"?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
.Ltmp11:
	jmp	.LBB11_3
.LBB11_3:
	leaq	32(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	jmp	.LBB11_8
.LBB11_7:                               # Block address taken
$ehgcr_11_7:
	jmp	.LBB11_8
.LBB11_8:
	jmp	.LBB11_9
.LBB11_9:
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	68(%rbp), %edx
.Ltmp14:
	xorl	%eax, %eax
	movb	%al, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp15:
	jmp	.LBB11_10
.LBB11_10:
	leaq	48(%rbp), %rcx
	callq	"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	movq	-24(%rbp), %rax                 # 8-byte Reload
	addq	$224, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z")@IMGREL
	.section	.text,"xr",discard,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.seh_endproc
	.def	"?dtor$4@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA":
.seh_proc "?dtor$4@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"
.LBB11_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	32(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.seh_endproc
	.def	"?catch$5@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?catch$5@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA":
.seh_proc "?catch$5@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"
	.seh_handler __CxxFrameHandler3, @unwind, @except
.LBB11_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	128(%rdx), %rbp
	.seh_endprologue
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp12:
	movl	$4, %edx
	movb	$1, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp13:
	jmp	.LBB11_6
.LBB11_6:
	leaq	.LBB11_7(%rip), %rax
	addq	$64, %rsp
	popq	%rbp
	retq                                    # CATCHRET
	.seh_handlerdata
	.long	("$cppxdata$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z")@IMGREL
	.section	.text,"xr",discard,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.seh_endproc
	.def	"?dtor$11@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$11@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA":
.seh_proc "?dtor$11@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"
.LBB11_11:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	48(%rbp), %rcx
	callq	"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end1:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.p2align	2
"$cppxdata$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z":
	.long	429065506                       # MagicNumber
	.long	4                               # MaxState
	.long	("$stateUnwindMap$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z")@IMGREL # UnwindMap
	.long	1                               # NumTryBlocks
	.long	("$tryMap$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z")@IMGREL # TryBlockMap
	.long	5                               # IPMapEntries
	.long	("$ip2state$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z")@IMGREL # IPToStateXData
	.long	216                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z":
	.long	-1                              # ToState
	.long	"?dtor$11@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	0                               # Action
	.long	1                               # ToState
	.long	"?dtor$4@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	0                               # Action
"$tryMap$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z":
	.long	1                               # TryLow
	.long	2                               # TryHigh
	.long	3                               # CatchHigh
	.long	1                               # NumCatches
	.long	("$handlerMap$0$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z")@IMGREL # HandlerArray
"$handlerMap$0$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z":
	.long	64                              # Adjectives
	.long	0                               # Type
	.long	0                               # CatchObjOffset
	.long	"?catch$5@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"@IMGREL # Handler
	.long	88                              # ParentFrameOffset
"$ip2state$??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z":
	.long	.Lfunc_begin1@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp8@IMGREL+1                 # IP
	.long	2                               # ToState
	.long	.Ltmp14@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp15@IMGREL+1                # IP
	.long	-1                              # ToState
	.long	"?catch$5@?0???$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z@4HA"@IMGREL # IP
	.long	3                               # ToState
	.section	.text,"xr",discard,"??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
                                        # -- End function
	.def	"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
	.globl	"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z" # -- Begin function ??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z
	.p2align	4, 0x90
"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z": # @"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
.Lfunc_begin2:
.seh_proc "??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$96, %rsp
	.seh_stackalloc 96
	leaq	96(%rsp), %rbp
	.seh_setframe %rbp, 96
	.seh_endprologue
	movq	$-2, -8(%rbp)
	andb	$1, %r8b
	movb	%r8b, -9(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-32(%rbp), %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rdx
	callq	"??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
                                        # kill: def $rcx killed $rax
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	%rax, %rcx
	addq	$8, %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	movq	(%rax), %rcx
	movb	-9(%rbp), %dl
.Ltmp16:
	andb	$1, %dl
	callq	"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
.Ltmp17:
	movb	%al, -33(%rbp)                  # 1-byte Spill
	jmp	.LBB12_1
.LBB12_1:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movb	-33(%rbp), %dl                  # 1-byte Reload
	andb	$1, %dl
	movb	%dl, (%rcx)
	addq	$96, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z")@IMGREL
	.section	.text,"xr",discard,"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z@4HA":
.seh_proc "?dtor$2@?0???0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z@4HA"
.LBB12_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	96(%rdx), %rbp
	.seh_endprologue
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	callq	"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end2:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
	.p2align	2
"$cppxdata$??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z")@IMGREL # IPToStateXData
	.long	88                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z@4HA"@IMGREL # Action
"$ip2state$??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z":
	.long	.Lfunc_begin2@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp16@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp17@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@_N@Z"
                                        # -- End function
	.def	"??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	.globl	"??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ" # -- Begin function ??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ
	.p2align	4, 0x90
"??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ": # @"??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
.seh_proc "??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	8(%rax), %al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.globl	"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z" # -- Begin function ??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z
	.p2align	4, 0x90
"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z": # @"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
.Lfunc_begin3:
.seh_proc "??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$128, %rsp
	.seh_stackalloc 128
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	leaq	-24(%rbp), %rcx
	xorl	%edx, %edx
	callq	"??0_Lockit@std@@QEAA@H@Z"
	movq	"?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"(%rip), %rax
	movq	%rax, -32(%rbp)
	leaq	"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"(%rip), %rcx
	callq	"??Bid@locale@std@@QEAA_KXZ"
	movq	%rax, -40(%rbp)
	movq	-16(%rbp), %rcx
	movq	-40(%rbp), %rdx
.Ltmp18:
	callq	"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
.Ltmp19:
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	jmp	.LBB14_1
.LBB14_1:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -48(%rbp)
	cmpq	$0, -48(%rbp)
	jne	.LBB14_12
# %bb.2:
	cmpq	$0, -32(%rbp)
	je	.LBB14_4
# %bb.3:
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	jmp	.LBB14_11
.LBB14_4:
	movq	-16(%rbp), %rdx
.Ltmp20:
	leaq	-32(%rbp), %rcx
	callq	"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Ltmp21:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB14_5
.LBB14_5:
	movq	-80(%rbp), %rax                 # 8-byte Reload
	cmpq	$-1, %rax
	jne	.LBB14_8
# %bb.6:
.Ltmp24:
	callq	"?_Throw_bad_cast@std@@YAXXZ"
.Ltmp25:
	jmp	.LBB14_7
.LBB14_7:
.LBB14_8:
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rdx
	leaq	-64(%rbp), %rcx
	callq	"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
	movq	-56(%rbp), %rcx
.Ltmp22:
	callq	"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"
.Ltmp23:
	jmp	.LBB14_9
.LBB14_9:
	movq	-56(%rbp), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, "?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	leaq	-64(%rbp), %rcx
	callq	"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	jmp	.LBB14_11
.LBB14_11:
	jmp	.LBB14_12
.LBB14_12:
	movq	-48(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z")@IMGREL
	.section	.text,"xr",discard,"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$10@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$10@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"
.LBB14_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$13@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$13@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$13@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"
.LBB14_13:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end3:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.p2align	2
"$cppxdata$??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z")@IMGREL # IPToStateXData
	.long	120                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z":
	.long	-1                              # ToState
	.long	"?dtor$13@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$10@?0???$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
"$ip2state$??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z":
	.long	.Lfunc_begin3@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp18@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp22@IMGREL+1                # IP
	.long	1                               # ToState
	.long	.Ltmp23@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
                                        # -- End function
	.def	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	.globl	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ" # -- Begin function ?getloc@ios_base@std@@QEBA?AVlocale@2@XZ
	.p2align	4, 0x90
"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ": # @"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
.seh_proc "?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rax
	movq	64(%rax), %rdx
	callq	"??0locale@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.globl	"?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z" # -- Begin function ?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z
	.p2align	4, 0x90
"?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z": # @"?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
.seh_proc "?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$112, %rsp
	.seh_stackalloc 112
	.seh_endprologue
	movq	%rdx, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	176(%rsp), %rax
	movq	168(%rsp), %rax
	movq	160(%rsp), %rax
	movq	%rdx, 104(%rsp)
	movq	%rcx, 96(%rsp)
	movq	96(%rsp), %rcx
	movq	176(%rsp), %r10
	movq	168(%rsp), %r11
	movq	160(%rsp), %rsi
	movq	(%r9), %rax
	movq	%rax, 80(%rsp)
	movq	8(%r9), %rax
	movq	%rax, 88(%rsp)
	movq	(%r8), %rax
	movq	%rax, 64(%rsp)
	movq	8(%r8), %rax
	movq	%rax, 72(%rsp)
	movq	(%rcx), %rax
	leaq	64(%rsp), %r8
	leaq	80(%rsp), %r9
	movq	%rsi, 32(%rsp)
	movq	%r11, 40(%rsp)
	movq	%r10, 48(%rsp)
	callq	*80(%rax)
	movq	56(%rsp), %rax                  # 8-byte Reload
	addq	$112, %rsp
	popq	%rsi
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.globl	"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ" # -- Begin function ??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ": # @"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ"
.seh_proc "??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	$0, (%rax)
	movb	$1, 8(%rax)
	movb	$0, 9(%rax)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z"
	.globl	"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z" # -- Begin function ??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z
	.p2align	4, 0x90
"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z": # @"??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z"
.seh_proc "??0?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@AEAV?$basic_istream@DU?$char_traits@D@std@@@1@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	$0, %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, %rcx
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	%rcx, (%rax)
	cmpq	$0, (%rax)
	setne	%cl
	xorb	$-1, %cl
	andb	$1, %cl
	movb	%cl, 8(%rax)
	movb	$0, 9(%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1locale@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1locale@std@@QEAA@XZ"
	.globl	"??1locale@std@@QEAA@XZ"        # -- Begin function ??1locale@std@@QEAA@XZ
	.p2align	4, 0x90
"??1locale@std@@QEAA@XZ":               # @"??1locale@std@@QEAA@XZ"
.seh_proc "??1locale@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	$0, 8(%rax)
	je	.LBB19_4
# %bb.1:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	8(%rax), %rcx
	movq	(%rcx), %rax
	callq	*16(%rax)
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB19_3
# %bb.2:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	(%rcx), %rax
	movl	$1, %edx
	callq	*(%rax)
.LBB19_3:
	jmp	.LBB19_4
.LBB19_4:
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	.globl	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z" # -- Begin function ?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z
	.p2align	4, 0x90
"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z": # @"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.seh_proc "?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	andb	$1, %r8b
	movb	%r8b, 71(%rsp)
	movl	%edx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movb	71(%rsp), %al
	movb	%al, 47(%rsp)                   # 1-byte Spill
	callq	"?rdstate@ios_base@std@@QEBAHXZ"
	movb	47(%rsp), %r8b                  # 1-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movl	%eax, %edx
	orl	64(%rsp), %edx
	andb	$1, %r8b
	callq	"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.globl	"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ" # -- Begin function ??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ": # @"??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
.seh_proc "??1sentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.globl	"??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z" # -- Begin function ??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z
	.p2align	4, 0x90
"??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z": # @"??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
.seh_proc "??0_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, (%rax)
	movq	(%rax), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	$0, %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, 40(%rsp)
	cmpq	$0, 40(%rsp)
	je	.LBB22_2
# %bb.1:
	movq	40(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
.LBB22_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.globl	"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z" # -- Begin function ?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z
	.p2align	4, 0x90
"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z": # @"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
.Lfunc_begin4:
.seh_proc "?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$160, %rsp
	.seh_stackalloc 160
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 24(%rbp)
	andb	$1, %dl
	movb	%dl, 22(%rbp)
	movq	%rcx, 8(%rbp)
	movq	8(%rbp), %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?good@ios_base@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB23_2
# %bb.1:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	$2, %edx
	xorl	%r8d, %r8d
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	movb	$0, 23(%rbp)
	jmp	.LBB23_23
.LBB23_2:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, (%rbp)
	cmpq	$0, (%rbp)
	je	.LBB23_4
# %bb.3:
	movq	(%rbp), %rcx
	callq	"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
.LBB23_4:
	movb	$0, -1(%rbp)
	testb	$1, 22(%rbp)
	jne	.LBB23_20
# %bb.5:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$1, %eax
	cmpl	$0, %eax
	je	.LBB23_20
# %bb.6:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	leaq	-32(%rbp), %rdx
	movq	%rdx, -64(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-64(%rbp), %rcx                 # 8-byte Reload
.Ltmp26:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp27:
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	jmp	.LBB23_7
.LBB23_7:
	leaq	-32(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, -16(%rbp)
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, %rcx
.Ltmp28:
	callq	"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.Ltmp29:
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	jmp	.LBB23_8
.LBB23_8:
	movl	-68(%rbp), %eax                 # 4-byte Reload
	movl	%eax, -36(%rbp)
.LBB23_9:                               # =>This Inner Loop Header: Depth=1
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, -40(%rbp)
	leaq	-40(%rbp), %rcx
	leaq	-36(%rbp), %rdx
	callq	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	testb	$1, %al
	jne	.LBB23_10
	jmp	.LBB23_12
.LBB23_10:
	movb	$1, -1(%rbp)
	jmp	.LBB23_19
.LBB23_12:                              #   in Loop: Header=BB23_9 Depth=1
	movq	-16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	leaq	-36(%rbp), %rcx
	callq	"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z"
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movb	%al, %r8b
	movl	$72, %edx
	callq	"?is@?$ctype@D@std@@QEBA_NFD@Z"
	testb	$1, %al
	jne	.LBB23_14
# %bb.13:
	jmp	.LBB23_19
.LBB23_14:                              #   in Loop: Header=BB23_9 Depth=1
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, %rcx
.Ltmp30:
	callq	"?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.Ltmp31:
	movl	%eax, -84(%rbp)                 # 4-byte Spill
	jmp	.LBB23_18
.LBB23_16:                              # Block address taken
$ehgcr_23_16:
	jmp	.LBB23_17
.LBB23_17:
	jmp	.LBB23_20
.LBB23_18:                              #   in Loop: Header=BB23_9 Depth=1
	movl	-84(%rbp), %eax                 # 4-byte Reload
	movl	%eax, -36(%rbp)
	jmp	.LBB23_9
.LBB23_19:
	jmp	.LBB23_17
.LBB23_20:
	testb	$1, -1(%rbp)
	je	.LBB23_22
# %bb.21:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	$3, %edx
	xorl	%r8d, %r8d
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.LBB23_22:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?good@ios_base@std@@QEBA_NXZ"
	andb	$1, %al
	movb	%al, 23(%rbp)
.LBB23_23:
	movb	23(%rbp), %al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$160, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z")@IMGREL
	.section	.text,"xr",discard,"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.seh_endproc
	.def	"?dtor$11@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$11@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA":
.seh_proc "?dtor$11@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA"
.LBB23_11:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-32(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.seh_endproc
	.def	"?catch$15@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?catch$15@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA":
.seh_proc "?catch$15@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA"
	.seh_handler __CxxFrameHandler3, @unwind, @except
.LBB23_15:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	$4, %edx
	movb	$1, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	leaq	.LBB23_16(%rip), %rax
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CATCHRET
.Lfunc_end4:
	.seh_handlerdata
	.long	("$cppxdata$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z")@IMGREL
	.section	.text,"xr",discard,"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.p2align	2
"$cppxdata$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z":
	.long	429065506                       # MagicNumber
	.long	3                               # MaxState
	.long	("$stateUnwindMap$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z")@IMGREL # UnwindMap
	.long	1                               # NumTryBlocks
	.long	("$tryMap$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z")@IMGREL # TryBlockMap
	.long	5                               # IPMapEntries
	.long	("$ip2state$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z")@IMGREL # IPToStateXData
	.long	152                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z":
	.long	-1                              # ToState
	.long	"?dtor$11@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	0                               # Action
	.long	-1                              # ToState
	.long	0                               # Action
"$tryMap$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z":
	.long	1                               # TryLow
	.long	1                               # TryHigh
	.long	2                               # CatchHigh
	.long	1                               # NumCatches
	.long	("$handlerMap$0$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z")@IMGREL # HandlerArray
"$handlerMap$0$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z":
	.long	64                              # Adjectives
	.long	0                               # Type
	.long	0                               # CatchObjOffset
	.long	"?catch$15@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA"@IMGREL # Handler
	.long	56                              # ParentFrameOffset
"$ip2state$?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z":
	.long	.Lfunc_begin4@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp26@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp28@IMGREL+1                # IP
	.long	1                               # ToState
	.long	.Ltmp31@IMGREL+1                # IP
	.long	-1                              # ToState
	.long	"?catch$15@?0??_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z@4HA"@IMGREL # IP
	.long	2                               # ToState
	.section	.text,"xr",discard,"?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
                                        # -- End function
	.def	"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.globl	"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ" # -- Begin function ??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ": # @"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
.Lfunc_begin5:
.seh_proc "??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, -24(%rbp)
	cmpq	$0, -24(%rbp)
	je	.LBB24_3
# %bb.1:
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rax
	movq	16(%rax), %rax
.Ltmp32:
	callq	*%rax
.Ltmp33:
	jmp	.LBB24_2
.LBB24_2:
	jmp	.LBB24_3
.LBB24_3:
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_endproc
	.def	"?dtor$4@?0???1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0???1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA":
.seh_proc "?dtor$4@?0???1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA"
.LBB24_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end5:
	.seh_handlerdata
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.p2align	2
"$cppxdata$??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	-1                              # ToState
	.long	"?dtor$4@?0???1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA"@IMGREL # Action
"$ip2state$??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	.Lfunc_begin5@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp32@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp33@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
                                        # -- End function
	.def	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	.globl	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ" # -- Begin function ?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ
	.p2align	4, 0x90
"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ": # @"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
.seh_proc "?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	72(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?good@ios_base@std@@QEBA_NXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?good@ios_base@std@@QEBA_NXZ"
	.globl	"?good@ios_base@std@@QEBA_NXZ"  # -- Begin function ?good@ios_base@std@@QEBA_NXZ
	.p2align	4, 0x90
"?good@ios_base@std@@QEBA_NXZ":         # @"?good@ios_base@std@@QEBA_NXZ"
.seh_proc "?good@ios_base@std@@QEBA_NXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"?rdstate@ios_base@std@@QEBAHXZ"
	cmpl	$0, %eax
	sete	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"
	.globl	"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ" # -- Begin function ?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ
	.p2align	4, 0x90
"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ": # @"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"
.seh_proc "?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	80(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.globl	"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ" # -- Begin function ?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ
	.p2align	4, 0x90
"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ": # @"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
.Lfunc_begin6:
.seh_proc "?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$96, %rsp
	.seh_stackalloc 96
	leaq	96(%rsp), %rbp
	.seh_setframe %rbp, 96
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, -24(%rbp)
	cmpq	$0, -24(%rbp)
	je	.LBB28_13
# %bb.1:
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	leaq	-40(%rbp), %rcx
	callq	"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	leaq	-40(%rbp), %rcx
	callq	"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB28_2
	jmp	.LBB28_11
.LBB28_2:
	movl	$0, -44(%rbp)
	movq	-24(%rbp), %rcx
.Ltmp34:
	callq	"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.Ltmp35:
	movl	%eax, -60(%rbp)                 # 4-byte Spill
	jmp	.LBB28_8
.LBB28_5:                               # Block address taken
$ehgcr_28_5:
	jmp	.LBB28_6
.LBB28_6:
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	-44(%rbp), %edx
.Ltmp38:
	xorl	%eax, %eax
	movb	%al, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp39:
	jmp	.LBB28_7
.LBB28_7:
	jmp	.LBB28_11
.LBB28_8:
	movl	-60(%rbp), %eax                 # 4-byte Reload
	cmpl	$-1, %eax
	jne	.LBB28_10
# %bb.9:
	movl	-44(%rbp), %eax
	orl	$4, %eax
	movl	%eax, -44(%rbp)
.LBB28_10:
	jmp	.LBB28_6
.LBB28_11:
	leaq	-40(%rbp), %rcx
	callq	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	jmp	.LBB28_13
.LBB28_13:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$96, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ")@IMGREL
	.section	.text,"xr",discard,"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.seh_endproc
	.def	"?catch$3@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?catch$3@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA":
.seh_proc "?catch$3@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA"
	.seh_handler __CxxFrameHandler3, @unwind, @except
.LBB28_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	96(%rdx), %rbp
	.seh_endprologue
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp36:
	movl	$4, %edx
	movb	$1, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp37:
	jmp	.LBB28_4
.LBB28_4:
	leaq	.LBB28_5(%rip), %rax
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CATCHRET
	.seh_handlerdata
	.long	("$cppxdata$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ")@IMGREL
	.section	.text,"xr",discard,"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.seh_endproc
	.def	"?dtor$12@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$12@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA":
.seh_proc "?dtor$12@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA"
.LBB28_12:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	96(%rdx), %rbp
	.seh_endprologue
	leaq	-40(%rbp), %rcx
	callq	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end6:
	.seh_handlerdata
	.section	.text,"xr",discard,"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.p2align	2
"$cppxdata$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ":
	.long	429065506                       # MagicNumber
	.long	3                               # MaxState
	.long	("$stateUnwindMap$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ")@IMGREL # UnwindMap
	.long	1                               # NumTryBlocks
	.long	("$tryMap$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ")@IMGREL # TryBlockMap
	.long	5                               # IPMapEntries
	.long	("$ip2state$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ")@IMGREL # IPToStateXData
	.long	88                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ":
	.long	-1                              # ToState
	.long	"?dtor$12@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	0                               # Action
	.long	0                               # ToState
	.long	0                               # Action
"$tryMap$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ":
	.long	1                               # TryLow
	.long	1                               # TryHigh
	.long	2                               # CatchHigh
	.long	1                               # NumCatches
	.long	("$handlerMap$0$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ")@IMGREL # HandlerArray
"$handlerMap$0$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ":
	.long	64                              # Adjectives
	.long	0                               # Type
	.long	0                               # CatchObjOffset
	.long	"?catch$3@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA"@IMGREL # Handler
	.long	56                              # ParentFrameOffset
"$ip2state$?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ":
	.long	.Lfunc_begin6@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp34@IMGREL+1                # IP
	.long	1                               # ToState
	.long	.Ltmp38@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp39@IMGREL+1                # IP
	.long	-1                              # ToState
	.long	"?catch$3@?0??flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ@4HA"@IMGREL # IP
	.long	2                               # ToState
	.section	.text,"xr",discard,"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
                                        # -- End function
	.def	"?flags@ios_base@std@@QEBAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?flags@ios_base@std@@QEBAHXZ"
	.globl	"?flags@ios_base@std@@QEBAHXZ"  # -- Begin function ?flags@ios_base@std@@QEBAHXZ
	.p2align	4, 0x90
"?flags@ios_base@std@@QEBAHXZ":         # @"?flags@ios_base@std@@QEBAHXZ"
.seh_proc "?flags@ios_base@std@@QEBAHXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	24(%rax), %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.globl	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z" # -- Begin function ??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z
	.p2align	4, 0x90
"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z": # @"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Lfunc_begin7:
.seh_proc "??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$128, %rsp
	.seh_stackalloc 128
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	leaq	-24(%rbp), %rcx
	xorl	%edx, %edx
	callq	"??0_Lockit@std@@QEAA@H@Z"
	movq	"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB"(%rip), %rax
	movq	%rax, -32(%rbp)
	leaq	"?id@?$ctype@D@std@@2V0locale@2@A"(%rip), %rcx
	callq	"??Bid@locale@std@@QEAA_KXZ"
	movq	%rax, -40(%rbp)
	movq	-16(%rbp), %rcx
	movq	-40(%rbp), %rdx
.Ltmp40:
	callq	"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
.Ltmp41:
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	jmp	.LBB30_1
.LBB30_1:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -48(%rbp)
	cmpq	$0, -48(%rbp)
	jne	.LBB30_12
# %bb.2:
	cmpq	$0, -32(%rbp)
	je	.LBB30_4
# %bb.3:
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	jmp	.LBB30_11
.LBB30_4:
	movq	-16(%rbp), %rdx
.Ltmp42:
	leaq	-32(%rbp), %rcx
	callq	"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Ltmp43:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB30_5
.LBB30_5:
	movq	-80(%rbp), %rax                 # 8-byte Reload
	cmpq	$-1, %rax
	jne	.LBB30_8
# %bb.6:
.Ltmp46:
	callq	"?_Throw_bad_cast@std@@YAXXZ"
.Ltmp47:
	jmp	.LBB30_7
.LBB30_7:
.LBB30_8:
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rdx
	leaq	-64(%rbp), %rcx
	callq	"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
	movq	-56(%rbp), %rcx
.Ltmp44:
	callq	"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"
.Ltmp45:
	jmp	.LBB30_9
.LBB30_9:
	movq	-56(%rbp), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, "?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB"(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	leaq	-64(%rbp), %rcx
	callq	"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	jmp	.LBB30_11
.LBB30_11:
	jmp	.LBB30_12
.LBB30_12:
	movq	-48(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z")@IMGREL
	.section	.text,"xr",discard,"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$10@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$10@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA"
.LBB30_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$13@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$13@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$13@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA"
.LBB30_13:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end7:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.p2align	2
"$cppxdata$??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z")@IMGREL # IPToStateXData
	.long	120                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z":
	.long	-1                              # ToState
	.long	"?dtor$13@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$10@?0???$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
"$ip2state$??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z":
	.long	.Lfunc_begin7@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp40@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp44@IMGREL+1                # IP
	.long	1                               # ToState
	.long	.Ltmp45@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
                                        # -- End function
	.def	"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.globl	"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ" # -- Begin function ?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ
	.p2align	4, 0x90
"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ": # @"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.seh_proc "?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	movq	%rax, %rcx
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	%rcx, %rax
	jge	.LBB31_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ"
	movq	%rax, %rcx
	callq	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	movl	%eax, 36(%rsp)                  # 4-byte Spill
	jmp	.LBB31_3
.LBB31_2:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	(%rcx), %rax
	callq	*48(%rax)
	movl	%eax, 36(%rsp)                  # 4-byte Spill
.LBB31_3:
	movl	36(%rsp), %eax                  # 4-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	.globl	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z" # -- Begin function ?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z
	.p2align	4, 0x90
"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z": # @"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
.seh_proc "?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	(%rax), %eax
	movq	8(%rsp), %rcx
	cmpl	(%rcx), %eax
	sete	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	.globl	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ" # -- Begin function ?eof@?$_Narrow_char_traits@DH@std@@SAHXZ
	.p2align	4, 0x90
"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ": # @"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
# %bb.0:
	movl	$4294967295, %eax               # imm = 0xFFFFFFFF
	retq
                                        # -- End function
	.def	"?is@?$ctype@D@std@@QEBA_NFD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?is@?$ctype@D@std@@QEBA_NFD@Z"
	.globl	"?is@?$ctype@D@std@@QEBA_NFD@Z" # -- Begin function ?is@?$ctype@D@std@@QEBA_NFD@Z
	.p2align	4, 0x90
"?is@?$ctype@D@std@@QEBA_NFD@Z":        # @"?is@?$ctype@D@std@@QEBA_NFD@Z"
.seh_proc "?is@?$ctype@D@std@@QEBA_NFD@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movb	%r8b, 15(%rsp)
	movw	%dx, 12(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	24(%rax), %rax
	movzbl	15(%rsp), %ecx
                                        # kill: def $rcx killed $ecx
	movswl	(%rax,%rcx,2), %eax
	movswl	12(%rsp), %ecx
	andl	%ecx, %eax
	cmpl	$0, %eax
	setne	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z"
	.globl	"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z" # -- Begin function ?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z
	.p2align	4, 0x90
"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z": # @"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z"
.seh_proc "?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	(%rax), %eax
                                        # kill: def $al killed $al killed $eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.globl	"?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ" # -- Begin function ?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ
	.p2align	4, 0x90
"?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ": # @"?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.seh_proc "?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	callq	"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	movq	%rax, %rcx
	movl	$1, %eax
	cmpq	%rcx, %rax
	jge	.LBB36_2
# %bb.1:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	movq	%rax, %rcx
	callq	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	jmp	.LBB36_6
.LBB36_2:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	movl	%eax, 60(%rsp)
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, 56(%rsp)
	leaq	56(%rsp), %rcx
	leaq	60(%rsp), %rdx
	callq	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	testb	$1, %al
	jne	.LBB36_3
	jmp	.LBB36_4
.LBB36_3:
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, 40(%rsp)                  # 4-byte Spill
	jmp	.LBB36_5
.LBB36_4:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	movl	%eax, 40(%rsp)                  # 4-byte Spill
.LBB36_5:
	movl	40(%rsp), %eax                  # 4-byte Reload
	movl	%eax, 44(%rsp)                  # 4-byte Spill
.LBB36_6:
	movl	44(%rsp), %eax                  # 4-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?rdstate@ios_base@std@@QEBAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?rdstate@ios_base@std@@QEBAHXZ"
	.globl	"?rdstate@ios_base@std@@QEBAHXZ" # -- Begin function ?rdstate@ios_base@std@@QEBAHXZ
	.p2align	4, 0x90
"?rdstate@ios_base@std@@QEBAHXZ":       # @"?rdstate@ios_base@std@@QEBAHXZ"
.seh_proc "?rdstate@ios_base@std@@QEBAHXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	16(%rax), %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.globl	"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z" # -- Begin function ??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z
	.p2align	4, 0x90
"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z": # @"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
.Lfunc_begin8:
.seh_proc "??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$96, %rsp
	.seh_stackalloc 96
	leaq	96(%rsp), %rbp
	.seh_setframe %rbp, 96
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-32(%rbp), %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	movq	%rcx, -16(%rbp)
	movq	-24(%rbp), %rdx
	callq	"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp48:
	callq	"?good@ios_base@std@@QEBA_NXZ"
.Ltmp49:
	movb	%al, -41(%rbp)                  # 1-byte Spill
	jmp	.LBB38_1
.LBB38_1:
	movb	-41(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB38_3
	jmp	.LBB38_2
.LBB38_2:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movb	$0, 8(%rax)
	jmp	.LBB38_9
.LBB38_3:
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, -40(%rbp)
	cmpq	$0, -40(%rbp)
	je	.LBB38_5
# %bb.4:
	movq	-40(%rbp), %rax
	cmpq	-24(%rbp), %rax
	jne	.LBB38_6
.LBB38_5:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movb	$1, 8(%rax)
	jmp	.LBB38_9
.LBB38_6:
	movq	-40(%rbp), %rcx
.Ltmp50:
	callq	"?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
.Ltmp51:
	jmp	.LBB38_7
.LBB38_7:
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp52:
	callq	"?good@ios_base@std@@QEBA_NXZ"
.Ltmp53:
	movb	%al, -57(%rbp)                  # 1-byte Spill
	jmp	.LBB38_8
.LBB38_8:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movb	-57(%rbp), %cl                  # 1-byte Reload
	andb	$1, %cl
	movb	%cl, 8(%rax)
.LBB38_9:
	movq	-16(%rbp), %rax
	addq	$96, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z")@IMGREL
	.section	.text,"xr",discard,"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.seh_endproc
	.def	"?dtor$10@?0???0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0???0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z@4HA":
.seh_proc "?dtor$10@?0???0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z@4HA"
.LBB38_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	96(%rdx), %rbp
	.seh_endprologue
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	callq	"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end8:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.p2align	2
"$cppxdata$??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z")@IMGREL # IPToStateXData
	.long	88                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z":
	.long	-1                              # ToState
	.long	"?dtor$10@?0???0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z@4HA"@IMGREL # Action
"$ip2state$??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z":
	.long	.Lfunc_begin8@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp48@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp53@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
                                        # -- End function
	.def	"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	.globl	"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ" # -- Begin function ??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ
	.p2align	4, 0x90
"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ": # @"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
.seh_proc "??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	8(%rax), %al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.globl	"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ" # -- Begin function ?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ
	.p2align	4, 0x90
"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ": # @"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.seh_proc "?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*104(%rax)
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.globl	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ" # -- Begin function ??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ": # @"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
.Lfunc_begin9:
.seh_proc "??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	callq	"?uncaught_exception@std@@YA_NXZ"
	xorb	$-1, %al
	andb	$1, %al
	movb	%al, -17(%rbp)
	testb	$1, -17(%rbp)
	je	.LBB41_3
# %bb.1:
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	(%rax), %rcx
.Ltmp54:
	callq	"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
.Ltmp55:
	jmp	.LBB41_2
.LBB41_2:
	jmp	.LBB41_3
.LBB41_3:
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	callq	"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL
	.section	.text,"xr",discard,"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_endproc
	.def	"?dtor$4@?0???1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0???1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA":
.seh_proc "?dtor$4@?0???1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA"
.LBB41_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end9:
	.seh_handlerdata
	.section	.text,"xr",discard,"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.p2align	2
"$cppxdata$??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	-1                              # ToState
	.long	"?dtor$4@?0???1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA"@IMGREL # Action
"$ip2state$??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	.Lfunc_begin9@IMGREL            # IP
	.long	-1                              # ToState
	.long	.Ltmp54@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp55@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
                                        # -- End function
	.def	"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	.globl	"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z" # -- Begin function ??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z
	.p2align	4, 0x90
"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z": # @"??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
.seh_proc "??0_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 64(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, (%rax)
	movq	(%rax), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	$0, %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, 40(%rsp)
	cmpq	$0, 40(%rsp)
	je	.LBB42_2
# %bb.1:
	movq	40(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
.LBB42_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.globl	"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ" # -- Begin function ??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ": # @"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
.Lfunc_begin10:
.seh_proc "??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	(%rax), %rcx
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, -24(%rbp)
	cmpq	$0, -24(%rbp)
	je	.LBB43_3
# %bb.1:
	movq	-24(%rbp), %rcx
	movq	(%rcx), %rax
	movq	16(%rax), %rax
.Ltmp56:
	callq	*%rax
.Ltmp57:
	jmp	.LBB43_2
.LBB43_2:
	jmp	.LBB43_3
.LBB43_3:
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_endproc
	.def	"?dtor$4@?0???1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0???1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA":
.seh_proc "?dtor$4@?0???1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA"
.LBB43_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end10:
	.seh_handlerdata
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	.p2align	2
"$cppxdata$??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	-1                              # ToState
	.long	"?dtor$4@?0???1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ@4HA"@IMGREL # Action
"$ip2state$??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ":
	.long	.Lfunc_begin10@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp56@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp57@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??1_Sentry_base@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
                                        # -- End function
	.def	"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
	.globl	"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ" # -- Begin function ?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ
	.p2align	4, 0x90
"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ": # @"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
.Lfunc_begin11:
.seh_proc "?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, -32(%rbp)                 # 8-byte Spill
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp58:
	callq	"?good@ios_base@std@@QEBA_NXZ"
.Ltmp59:
	movb	%al, -17(%rbp)                  # 1-byte Spill
	jmp	.LBB44_1
.LBB44_1:
	movb	-17(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB44_2
	jmp	.LBB44_11
.LBB44_2:
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$2, %eax
	cmpl	$0, %eax
	je	.LBB44_11
# %bb.3:
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, %rcx
.Ltmp60:
	callq	"?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.Ltmp61:
	movl	%eax, -36(%rbp)                 # 4-byte Spill
	jmp	.LBB44_4
.LBB44_4:
	movl	-36(%rbp), %eax                 # 4-byte Reload
	cmpl	$-1, %eax
	jne	.LBB44_10
# %bb.5:
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp62:
	xorl	%eax, %eax
	movb	%al, %r8b
	movl	$4, %edx
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp63:
	jmp	.LBB44_9
.LBB44_7:                               # Block address taken
$ehgcr_44_7:
	jmp	.LBB44_8
.LBB44_8:
	addq	$80, %rsp
	popq	%rbp
	retq
.LBB44_9:
	jmp	.LBB44_10
.LBB44_10:
	jmp	.LBB44_11
.LBB44_11:
	jmp	.LBB44_8
	.seh_handlerdata
	.long	("$cppxdata$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ")@IMGREL
	.section	.text,"xr",discard,"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
	.seh_endproc
	.def	"?catch$6@?0??_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?catch$6@?0??_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ@4HA":
.seh_proc "?catch$6@?0??_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ@4HA"
	.seh_handler __CxxFrameHandler3, @unwind, @except
.LBB44_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	leaq	.LBB44_7(%rip), %rax
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CATCHRET
.Lfunc_end11:
	.seh_handlerdata
	.long	("$cppxdata$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ")@IMGREL
	.section	.text,"xr",discard,"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
	.p2align	2
"$cppxdata$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ")@IMGREL # UnwindMap
	.long	1                               # NumTryBlocks
	.long	("$tryMap$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ")@IMGREL # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ":
	.long	-1                              # ToState
	.long	0                               # Action
	.long	-1                              # ToState
	.long	0                               # Action
"$tryMap$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ":
	.long	0                               # TryLow
	.long	0                               # TryHigh
	.long	1                               # CatchHigh
	.long	1                               # NumCatches
	.long	("$handlerMap$0$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ")@IMGREL # HandlerArray
"$handlerMap$0$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ":
	.long	64                              # Adjectives
	.long	0                               # Type
	.long	0                               # CatchObjOffset
	.long	"?catch$6@?0??_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ@4HA"@IMGREL # Handler
	.long	56                              # ParentFrameOffset
"$ip2state$?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ":
	.long	.Lfunc_begin11@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp58@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp63@IMGREL+1                # IP
	.long	-1                              # ToState
	.long	"?catch$6@?0??_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ@4HA"@IMGREL # IP
	.long	1                               # ToState
	.section	.text,"xr",discard,"?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
                                        # -- End function
	.def	"??Bid@locale@std@@QEAA_KXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??Bid@locale@std@@QEAA_KXZ"
	.globl	"??Bid@locale@std@@QEAA_KXZ"    # -- Begin function ??Bid@locale@std@@QEAA_KXZ
	.p2align	4, 0x90
"??Bid@locale@std@@QEAA_KXZ":           # @"??Bid@locale@std@@QEAA_KXZ"
.seh_proc "??Bid@locale@std@@QEAA_KXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	cmpq	$0, (%rax)
	jne	.LBB45_4
# %bb.1:
	leaq	40(%rsp), %rcx
	xorl	%edx, %edx
	callq	"??0_Lockit@std@@QEAA@H@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	$0, (%rax)
	jne	.LBB45_3
# %bb.2:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movl	"?_Id_cnt@id@locale@std@@0HA"(%rip), %ecx
	addl	$1, %ecx
	movl	%ecx, "?_Id_cnt@id@locale@std@@0HA"(%rip)
	movslq	%ecx, %rcx
	movq	%rcx, (%rax)
.LBB45_3:
	leaq	40(%rsp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
.LBB45_4:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
	.globl	"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z" # -- Begin function ?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z
	.p2align	4, 0x90
"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z": # @"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
.seh_proc "?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %rax
	movq	8(%rcx), %rcx
	cmpq	24(%rcx), %rax
	jae	.LBB46_2
# %bb.1:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	8(%rax), %rax
	movq	16(%rax), %rax
	movq	72(%rsp), %rcx
	movq	(%rax,%rcx,8), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	jmp	.LBB46_3
.LBB46_2:
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	jmp	.LBB46_3
.LBB46_3:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 56(%rsp)
	cmpq	$0, 56(%rsp)
	jne	.LBB46_5
# %bb.4:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	8(%rax), %rax
	testb	$1, 36(%rax)
	jne	.LBB46_6
.LBB46_5:
	movq	56(%rsp), %rax
	movq	%rax, 80(%rsp)
	jmp	.LBB46_9
.LBB46_6:
	callq	"?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ"
	movq	%rax, 48(%rsp)
	movq	72(%rsp), %rax
	movq	48(%rsp), %rcx
	cmpq	24(%rcx), %rax
	jae	.LBB46_8
# %bb.7:
	movq	48(%rsp), %rax
	movq	16(%rax), %rax
	movq	72(%rsp), %rcx
	movq	(%rax,%rcx,8), %rax
	movq	%rax, 80(%rsp)
	jmp	.LBB46_9
.LBB46_8:
	movq	$0, 80(%rsp)
.LBB46_9:
	movq	80(%rsp), %rax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.globl	"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z" # -- Begin function ?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z
	.p2align	4, 0x90
"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z": # @"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Lfunc_begin12:
.seh_proc "?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$176, %rsp
	.seh_stackalloc 176
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 40(%rbp)
	movq	%rdx, 32(%rbp)
	movq	%rcx, 24(%rbp)
	cmpq	$0, 24(%rbp)
	je	.LBB47_9
# %bb.1:
	movq	24(%rbp), %rax
	cmpq	$0, (%rax)
	jne	.LBB47_9
# %bb.2:
	movl	$48, %ecx
	callq	"??2@YAPEAX_K@Z"
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movb	$1, -81(%rbp)
	movq	32(%rbp), %rcx
	callq	"?_C_str@locale@std@@QEBAPEBDXZ"
	movq	%rax, %rdx
.Ltmp64:
	leaq	-80(%rbp), %rcx
	callq	"??0_Locinfo@std@@QEAA@PEBD@Z"
.Ltmp65:
	jmp	.LBB47_3
.LBB47_3:
.Ltmp66:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	xorl	%eax, %eax
	movl	%eax, %r8d
	leaq	-80(%rbp), %rdx
	callq	"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
.Ltmp67:
	jmp	.LBB47_4
.LBB47_4:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	movb	$0, -81(%rbp)
	movq	24(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	jmp	.LBB47_9
.LBB47_9:
	movl	$2, %eax
	addq	$176, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$5@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$5@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$5@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB47_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$6@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$6@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB47_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	testb	$1, -81(%rbp)
	jne	.LBB47_7
	jmp	.LBB47_8
.LBB47_7:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB47_8:
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end12:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.p2align	2
"$cppxdata$?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # IPToStateXData
	.long	168                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	-1                              # ToState
	.long	"?dtor$6@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$5@?0??_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
"$ip2state$?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	.Lfunc_begin12@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp64@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp66@IMGREL+1                # IP
	.long	1                               # ToState
	.long	.Ltmp67@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
                                        # -- End function
	.def	"?_Throw_bad_cast@std@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Throw_bad_cast@std@@YAXXZ"
	.globl	"?_Throw_bad_cast@std@@YAXXZ"   # -- Begin function ?_Throw_bad_cast@std@@YAXXZ
	.p2align	4, 0x90
"?_Throw_bad_cast@std@@YAXXZ":          # @"?_Throw_bad_cast@std@@YAXXZ"
.seh_proc "?_Throw_bad_cast@std@@YAXXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	leaq	32(%rsp), %rcx
	callq	"??0bad_cast@std@@QEAA@XZ"
	leaq	32(%rsp), %rcx
	leaq	"_TI2?AVbad_cast@std@@"(%rip), %rdx
	callq	_CxxThrowException
	int3
	.seh_endproc
                                        # -- End function
	.def	"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
	.globl	"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z" # -- Begin function ??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z
	.p2align	4, 0x90
"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z": # @"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
.seh_proc "??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movb	48(%rsp), %dl
	leaq	64(%rsp), %r8
	callq	"??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
	.globl	"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ" # -- Begin function ?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ
	.p2align	4, 0x90
"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ": # @"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
.seh_proc "?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	$0, 40(%rsp)
	leaq	40(%rsp), %rdx
	callq	"??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	.globl	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ" # -- Begin function ??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ": # @"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
.seh_proc "??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	$0, (%rax)
	je	.LBB51_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ"
	movq	%rax, %rcx
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rdx
	callq	"??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z"
.LBB51_2:
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_C_str@locale@std@@QEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_C_str@locale@std@@QEBAPEBDXZ"
	.globl	"?_C_str@locale@std@@QEBAPEBDXZ" # -- Begin function ?_C_str@locale@std@@QEBAPEBDXZ
	.p2align	4, 0x90
"?_C_str@locale@std@@QEBAPEBDXZ":       # @"?_C_str@locale@std@@QEBAPEBDXZ"
.seh_proc "?_C_str@locale@std@@QEBAPEBDXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	$0, 8(%rax)
	je	.LBB52_2
# %bb.1:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	8(%rax), %rcx
	addq	$40, %rcx
	callq	"?c_str@?$_Yarn@D@std@@QEBAPEBDXZ"
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	jmp	.LBB52_3
.LBB52_2:
	leaq	"??_C@_00CNPNBAHC@?$AA@"(%rip), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	jmp	.LBB52_3
.LBB52_3:
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_Locinfo@std@@QEAA@PEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.globl	"??0_Locinfo@std@@QEAA@PEBD@Z"  # -- Begin function ??0_Locinfo@std@@QEAA@PEBD@Z
	.p2align	4, 0x90
"??0_Locinfo@std@@QEAA@PEBD@Z":         # @"??0_Locinfo@std@@QEAA@PEBD@Z"
.Lfunc_begin13:
.seh_proc "??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$112, %rsp
	.seh_stackalloc 112
	leaq	112(%rsp), %rbp
	.seh_setframe %rbp, 112
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rcx
	movq	%rcx, -40(%rbp)                 # 8-byte Spill
	xorl	%edx, %edx
	callq	"??0_Lockit@std@@QEAA@H@Z"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	addq	$8, %rcx
	movq	%rcx, -80(%rbp)                 # 8-byte Spill
	callq	"??0?$_Yarn@D@std@@QEAA@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	addq	$24, %rcx
	movq	%rcx, -72(%rbp)                 # 8-byte Spill
	callq	"??0?$_Yarn@D@std@@QEAA@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	addq	$40, %rcx
	movq	%rcx, -64(%rbp)                 # 8-byte Spill
	callq	"??0?$_Yarn@_W@std@@QEAA@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	addq	$56, %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	callq	"??0?$_Yarn@_W@std@@QEAA@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	addq	$72, %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	callq	"??0?$_Yarn@D@std@@QEAA@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	addq	$88, %rcx
	movq	%rcx, -32(%rbp)                 # 8-byte Spill
	callq	"??0?$_Yarn@D@std@@QEAA@XZ"
	cmpq	$0, -16(%rbp)
	je	.LBB53_3
# %bb.1:
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	movq	-16(%rbp), %rdx
.Ltmp70:
	callq	"?_Locinfo_ctor@_Locinfo@std@@SAXPEAV12@PEBD@Z"
.Ltmp71:
	jmp	.LBB53_2
.LBB53_2:
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	retq
.LBB53_3:
.Ltmp68:
	leaq	"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@"(%rip), %rcx
	callq	"?_Xruntime_error@std@@YAXPEBD@Z"
.Ltmp69:
	jmp	.LBB53_4
.LBB53_4:
	int3
	.seh_handlerdata
	.long	("$cppxdata$??0_Locinfo@std@@QEAA@PEBD@Z")@IMGREL
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$5@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$5@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$5@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$6@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$6@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$7@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$7@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$7@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_7:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Yarn@_W@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$8@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$8@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$8@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_8:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Yarn@_W@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$9@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$9@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$9@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_9:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$10@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$10@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.def	"?dtor$11@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$11@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA":
.seh_proc "?dtor$11@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"
.LBB53_11:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	callq	"??1_Lockit@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end13:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0_Locinfo@std@@QEAA@PEBD@Z"
	.p2align	2
"$cppxdata$??0_Locinfo@std@@QEAA@PEBD@Z":
	.long	429065506                       # MagicNumber
	.long	7                               # MaxState
	.long	("$stateUnwindMap$??0_Locinfo@std@@QEAA@PEBD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0_Locinfo@std@@QEAA@PEBD@Z")@IMGREL # IPToStateXData
	.long	104                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0_Locinfo@std@@QEAA@PEBD@Z":
	.long	-1                              # ToState
	.long	"?dtor$11@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$10@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$9@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
	.long	2                               # ToState
	.long	"?dtor$8@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
	.long	3                               # ToState
	.long	"?dtor$7@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
	.long	4                               # ToState
	.long	"?dtor$6@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
	.long	5                               # ToState
	.long	"?dtor$5@?0???0_Locinfo@std@@QEAA@PEBD@Z@4HA"@IMGREL # Action
"$ip2state$??0_Locinfo@std@@QEAA@PEBD@Z":
	.long	.Lfunc_begin13@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp70@IMGREL+1                # IP
	.long	6                               # ToState
	.long	.Ltmp69@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0_Locinfo@std@@QEAA@PEBD@Z"
                                        # -- End function
	.def	"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.globl	"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z" # -- Begin function ??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z
	.p2align	4, 0x90
"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z": # @"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
.Lfunc_begin14:
.seh_proc "??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%r8, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-32(%rbp), %rcx
	movq	%rcx, -40(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rdx
	callq	"??0ctype_base@std@@QEAA@_K@Z"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	leaq	"??_7?$ctype@D@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	movq	-24(%rbp), %rdx
.Ltmp72:
	callq	"?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z"
.Ltmp73:
	jmp	.LBB54_1
.LBB54_1:
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z")@IMGREL
	.section	.text,"xr",discard,"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z@4HA":
.seh_proc "?dtor$2@?0???0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z@4HA"
.LBB54_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	callq	"??1ctype_base@std@@UEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end14:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.p2align	2
"$cppxdata$??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z@4HA"@IMGREL # Action
"$ip2state$??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z":
	.long	.Lfunc_begin14@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp72@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp73@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0?$ctype@D@std@@QEAA@AEBV_Locinfo@1@_K@Z"
                                        # -- End function
	.def	"??1_Locinfo@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_Locinfo@std@@QEAA@XZ"
	.globl	"??1_Locinfo@std@@QEAA@XZ"      # -- Begin function ??1_Locinfo@std@@QEAA@XZ
	.p2align	4, 0x90
"??1_Locinfo@std@@QEAA@XZ":             # @"??1_Locinfo@std@@QEAA@XZ"
.Lfunc_begin15:
.seh_proc "??1_Locinfo@std@@QEAA@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, -24(%rbp)                 # 8-byte Spill
.Ltmp74:
	callq	"?_Locinfo_dtor@_Locinfo@std@@SAXPEAV12@@Z"
.Ltmp75:
	jmp	.LBB55_1
.LBB55_1:
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	addq	$88, %rcx
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	addq	$72, %rcx
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	addq	$56, %rcx
	callq	"??1?$_Yarn@_W@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	addq	$40, %rcx
	callq	"??1?$_Yarn@_W@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	addq	$24, %rcx
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	addq	$8, %rcx
	callq	"??1?$_Yarn@D@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
	callq	"??1_Lockit@std@@QEAA@XZ"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??1_Locinfo@std@@QEAA@XZ")@IMGREL
	.section	.text,"xr",discard,"??1_Locinfo@std@@QEAA@XZ"
	.seh_endproc
	.def	"?dtor$2@?0???1_Locinfo@std@@QEAA@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???1_Locinfo@std@@QEAA@XZ@4HA":
.seh_proc "?dtor$2@?0???1_Locinfo@std@@QEAA@XZ@4HA"
.LBB55_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end15:
	.seh_handlerdata
	.section	.text,"xr",discard,"??1_Locinfo@std@@QEAA@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"??1_Locinfo@std@@QEAA@XZ"
	.p2align	2
"$cppxdata$??1_Locinfo@std@@QEAA@XZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??1_Locinfo@std@@QEAA@XZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??1_Locinfo@std@@QEAA@XZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??1_Locinfo@std@@QEAA@XZ":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???1_Locinfo@std@@QEAA@XZ@4HA"@IMGREL # Action
"$ip2state$??1_Locinfo@std@@QEAA@XZ":
	.long	.Lfunc_begin15@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp74@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp75@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??1_Locinfo@std@@QEAA@XZ"
                                        # -- End function
	.def	"?c_str@?$_Yarn@D@std@@QEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?c_str@?$_Yarn@D@std@@QEBAPEBDXZ"
	.globl	"?c_str@?$_Yarn@D@std@@QEBAPEBDXZ" # -- Begin function ?c_str@?$_Yarn@D@std@@QEBAPEBDXZ
	.p2align	4, 0x90
"?c_str@?$_Yarn@D@std@@QEBAPEBDXZ":     # @"?c_str@?$_Yarn@D@std@@QEBAPEBDXZ"
.seh_proc "?c_str@?$_Yarn@D@std@@QEBAPEBDXZ"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	cmpq	$0, (%rax)
	je	.LBB56_2
# %bb.1:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	(%rax), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	jmp	.LBB56_3
.LBB56_2:
	movq	8(%rsp), %rax                   # 8-byte Reload
	addq	$8, %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
.LBB56_3:
	movq	(%rsp), %rax                    # 8-byte Reload
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$_Yarn@D@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$_Yarn@D@std@@QEAA@XZ"
	.globl	"??0?$_Yarn@D@std@@QEAA@XZ"     # -- Begin function ??0?$_Yarn@D@std@@QEAA@XZ
	.p2align	4, 0x90
"??0?$_Yarn@D@std@@QEAA@XZ":            # @"??0?$_Yarn@D@std@@QEAA@XZ"
.seh_proc "??0?$_Yarn@D@std@@QEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	$0, (%rax)
	movb	$0, 8(%rax)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$_Yarn@_W@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$_Yarn@_W@std@@QEAA@XZ"
	.globl	"??0?$_Yarn@_W@std@@QEAA@XZ"    # -- Begin function ??0?$_Yarn@_W@std@@QEAA@XZ
	.p2align	4, 0x90
"??0?$_Yarn@_W@std@@QEAA@XZ":           # @"??0?$_Yarn@_W@std@@QEAA@XZ"
.seh_proc "??0?$_Yarn@_W@std@@QEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	$0, (%rax)
	movw	$0, 8(%rax)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$_Yarn@D@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$_Yarn@D@std@@QEAA@XZ"
	.globl	"??1?$_Yarn@D@std@@QEAA@XZ"     # -- Begin function ??1?$_Yarn@D@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$_Yarn@D@std@@QEAA@XZ":            # @"??1?$_Yarn@D@std@@QEAA@XZ"
.seh_proc "??1?$_Yarn@D@std@@QEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$_Yarn@_W@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$_Yarn@_W@std@@QEAA@XZ"
	.globl	"??1?$_Yarn@_W@std@@QEAA@XZ"    # -- Begin function ??1?$_Yarn@_W@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$_Yarn@_W@std@@QEAA@XZ":           # @"??1?$_Yarn@_W@std@@QEAA@XZ"
.seh_proc "??1?$_Yarn@_W@std@@QEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Tidy@?$_Yarn@D@std@@AEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	.globl	"?_Tidy@?$_Yarn@D@std@@AEAAXXZ" # -- Begin function ?_Tidy@?$_Yarn@D@std@@AEAAXXZ
	.p2align	4, 0x90
"?_Tidy@?$_Yarn@D@std@@AEAAXXZ":        # @"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
.Lfunc_begin16:
.seh_proc "?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	cmpq	$0, (%rax)
	je	.LBB61_3
# %bb.1:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	(%rax), %rcx
.Ltmp76:
	callq	free
.Ltmp77:
	jmp	.LBB61_2
.LBB61_2:
	jmp	.LBB61_3
.LBB61_3:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	$0, (%rax)
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Tidy@?$_Yarn@D@std@@AEAAXXZ")@IMGREL
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	.seh_endproc
	.def	"?dtor$4@?0??_Tidy@?$_Yarn@D@std@@AEAAXXZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0??_Tidy@?$_Yarn@D@std@@AEAAXXZ@4HA":
.seh_proc "?dtor$4@?0??_Tidy@?$_Yarn@D@std@@AEAAXXZ@4HA"
.LBB61_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end16:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	.p2align	2
"$cppxdata$?_Tidy@?$_Yarn@D@std@@AEAAXXZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?_Tidy@?$_Yarn@D@std@@AEAAXXZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?_Tidy@?$_Yarn@D@std@@AEAAXXZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Tidy@?$_Yarn@D@std@@AEAAXXZ":
	.long	-1                              # ToState
	.long	"?dtor$4@?0??_Tidy@?$_Yarn@D@std@@AEAAXXZ@4HA"@IMGREL # Action
"$ip2state$?_Tidy@?$_Yarn@D@std@@AEAAXXZ":
	.long	.Lfunc_begin16@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp76@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp77@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
                                        # -- End function
	.def	"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	.globl	"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ" # -- Begin function ?_Tidy@?$_Yarn@_W@std@@AEAAXXZ
	.p2align	4, 0x90
"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ":       # @"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
.Lfunc_begin17:
.seh_proc "?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	cmpq	$0, (%rax)
	je	.LBB62_3
# %bb.1:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	(%rax), %rcx
.Ltmp78:
	callq	free
.Ltmp79:
	jmp	.LBB62_2
.LBB62_2:
	jmp	.LBB62_3
.LBB62_3:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	$0, (%rax)
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Tidy@?$_Yarn@_W@std@@AEAAXXZ")@IMGREL
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	.seh_endproc
	.def	"?dtor$4@?0??_Tidy@?$_Yarn@_W@std@@AEAAXXZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0??_Tidy@?$_Yarn@_W@std@@AEAAXXZ@4HA":
.seh_proc "?dtor$4@?0??_Tidy@?$_Yarn@_W@std@@AEAAXXZ@4HA"
.LBB62_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end17:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	.p2align	2
"$cppxdata$?_Tidy@?$_Yarn@_W@std@@AEAAXXZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?_Tidy@?$_Yarn@_W@std@@AEAAXXZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?_Tidy@?$_Yarn@_W@std@@AEAAXXZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Tidy@?$_Yarn@_W@std@@AEAAXXZ":
	.long	-1                              # ToState
	.long	"?dtor$4@?0??_Tidy@?$_Yarn@_W@std@@AEAAXXZ@4HA"@IMGREL # Action
"$ip2state$?_Tidy@?$_Yarn@_W@std@@AEAAXXZ":
	.long	.Lfunc_begin17@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp78@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp79@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
                                        # -- End function
	.def	"??0ctype_base@std@@QEAA@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0ctype_base@std@@QEAA@_K@Z"
	.globl	"??0ctype_base@std@@QEAA@_K@Z"  # -- Begin function ??0ctype_base@std@@QEAA@_K@Z
	.p2align	4, 0x90
"??0ctype_base@std@@QEAA@_K@Z":         # @"??0ctype_base@std@@QEAA@_K@Z"
.seh_proc "??0ctype_base@std@@QEAA@_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0facet@locale@std@@IEAA@_K@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7ctype_base@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z"
	.globl	"?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z" # -- Begin function ?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z
	.p2align	4, 0x90
"?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z": # @"?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z"
.seh_proc "?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rdx, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	72(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	80(%rsp), %rcx
	leaq	40(%rsp), %rdx
	callq	"?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ"
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	40(%rsp), %rcx
	movq	%rcx, 16(%rax)
	movq	48(%rsp), %rcx
	movq	%rcx, 24(%rax)
	movq	56(%rsp), %rcx
	movq	%rcx, 32(%rax)
	movq	64(%rsp), %rcx
	movq	%rcx, 40(%rax)
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1ctype_base@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1ctype_base@std@@UEAA@XZ"
	.globl	"??1ctype_base@std@@UEAA@XZ"    # -- Begin function ??1ctype_base@std@@UEAA@XZ
	.p2align	4, 0x90
"??1ctype_base@std@@UEAA@XZ":           # @"??1ctype_base@std@@UEAA@XZ"
.seh_proc "??1ctype_base@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1facet@locale@std@@MEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_G?$ctype@D@std@@MEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G?$ctype@D@std@@MEAAPEAXI@Z"
	.globl	"??_G?$ctype@D@std@@MEAAPEAXI@Z" # -- Begin function ??_G?$ctype@D@std@@MEAAPEAXI@Z
	.p2align	4, 0x90
"??_G?$ctype@D@std@@MEAAPEAXI@Z":       # @"??_G?$ctype@D@std@@MEAAPEAXI@Z"
.seh_proc "??_G?$ctype@D@std@@MEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1?$ctype@D@std@@MEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB66_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB66_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Incref@facet@locale@std@@UEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Incref@facet@locale@std@@UEAAXXZ"
	.globl	"?_Incref@facet@locale@std@@UEAAXXZ" # -- Begin function ?_Incref@facet@locale@std@@UEAAXXZ
	.p2align	4, 0x90
"?_Incref@facet@locale@std@@UEAAXXZ":   # @"?_Incref@facet@locale@std@@UEAAXXZ"
.seh_proc "?_Incref@facet@locale@std@@UEAAXXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	lock		incl	8(%rax)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
	.globl	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ" # -- Begin function ?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ
	.p2align	4, 0x90
"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ": # @"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
.seh_proc "?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rcx, 8(%rsp)
	movq	8(%rsp), %rcx
	movq	%rcx, (%rsp)                    # 8-byte Spill
	movl	$-1, %eax
	lock		xaddl	%eax, 8(%rcx)
	subl	$1, %eax
	cmpl	$0, %eax
	jne	.LBB68_2
# %bb.1:
	movq	(%rsp), %rax                    # 8-byte Reload
	movq	%rax, 16(%rsp)
	jmp	.LBB68_3
.LBB68_2:
	movq	$0, 16(%rsp)
.LBB68_3:
	movq	16(%rsp), %rax
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
	.globl	"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z" # -- Begin function ?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z
	.p2align	4, 0x90
"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z": # @"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
.seh_proc "?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r8, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	leaq	56(%rsp), %rcx
	leaq	64(%rsp), %rdx
	callq	"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z"
.LBB69_1:                               # =>This Inner Loop Header: Depth=1
	movq	56(%rsp), %rax
	cmpq	64(%rsp), %rax
	je	.LBB69_4
# %bb.2:                                #   in Loop: Header=BB69_1 Depth=1
	movq	40(%rsp), %rdx                  # 8-byte Reload
	addq	$16, %rdx
	movq	56(%rsp), %rax
	movzbl	(%rax), %ecx
	callq	_Tolower
	movb	%al, %cl
	movq	56(%rsp), %rax
	movb	%cl, (%rax)
# %bb.3:                                #   in Loop: Header=BB69_1 Depth=1
	movq	56(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 56(%rsp)
	jmp	.LBB69_1
.LBB69_4:
	movq	56(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_tolower@?$ctype@D@std@@MEBADD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_tolower@?$ctype@D@std@@MEBADD@Z"
	.globl	"?do_tolower@?$ctype@D@std@@MEBADD@Z" # -- Begin function ?do_tolower@?$ctype@D@std@@MEBADD@Z
	.p2align	4, 0x90
"?do_tolower@?$ctype@D@std@@MEBADD@Z":  # @"?do_tolower@?$ctype@D@std@@MEBADD@Z"
.seh_proc "?do_tolower@?$ctype@D@std@@MEBADD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rdx
	addq	$16, %rdx
	movzbl	55(%rsp), %ecx
	callq	_Tolower
                                        # kill: def $al killed $al killed $eax
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
	.globl	"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z" # -- Begin function ?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z
	.p2align	4, 0x90
"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z": # @"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
.seh_proc "?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r8, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	leaq	56(%rsp), %rcx
	leaq	64(%rsp), %rdx
	callq	"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z"
.LBB71_1:                               # =>This Inner Loop Header: Depth=1
	movq	56(%rsp), %rax
	cmpq	64(%rsp), %rax
	je	.LBB71_4
# %bb.2:                                #   in Loop: Header=BB71_1 Depth=1
	movq	40(%rsp), %rdx                  # 8-byte Reload
	addq	$16, %rdx
	movq	56(%rsp), %rax
	movzbl	(%rax), %ecx
	callq	_Toupper
	movb	%al, %cl
	movq	56(%rsp), %rax
	movb	%cl, (%rax)
# %bb.3:                                #   in Loop: Header=BB71_1 Depth=1
	movq	56(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 56(%rsp)
	jmp	.LBB71_1
.LBB71_4:
	movq	56(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_toupper@?$ctype@D@std@@MEBADD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_toupper@?$ctype@D@std@@MEBADD@Z"
	.globl	"?do_toupper@?$ctype@D@std@@MEBADD@Z" # -- Begin function ?do_toupper@?$ctype@D@std@@MEBADD@Z
	.p2align	4, 0x90
"?do_toupper@?$ctype@D@std@@MEBADD@Z":  # @"?do_toupper@?$ctype@D@std@@MEBADD@Z"
.seh_proc "?do_toupper@?$ctype@D@std@@MEBADD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rdx
	addq	$16, %rdx
	movzbl	55(%rsp), %ecx
	callq	_Toupper
                                        # kill: def $al killed $al killed $eax
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z"
	.globl	"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z" # -- Begin function ?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z
	.p2align	4, 0x90
"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z": # @"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z"
.seh_proc "?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r9, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	leaq	48(%rsp), %rcx
	leaq	56(%rsp), %rdx
	callq	"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z"
	movq	64(%rsp), %rcx
	movq	48(%rsp), %rdx
	movq	56(%rsp), %r8
	movq	48(%rsp), %rax
	subq	%rax, %r8
	callq	memcpy
	movq	56(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_widen@?$ctype@D@std@@MEBADD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_widen@?$ctype@D@std@@MEBADD@Z"
	.globl	"?do_widen@?$ctype@D@std@@MEBADD@Z" # -- Begin function ?do_widen@?$ctype@D@std@@MEBADD@Z
	.p2align	4, 0x90
"?do_widen@?$ctype@D@std@@MEBADD@Z":    # @"?do_widen@?$ctype@D@std@@MEBADD@Z"
.seh_proc "?do_widen@?$ctype@D@std@@MEBADD@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movb	%dl, 15(%rsp)
	movq	%rcx, (%rsp)
	movb	15(%rsp), %al
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z"
	.globl	"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z" # -- Begin function ?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z
	.p2align	4, 0x90
"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z": # @"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z"
.seh_proc "?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	112(%rsp), %rax
	movb	%r9b, 71(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	leaq	48(%rsp), %rcx
	leaq	56(%rsp), %rdx
	callq	"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z"
	movq	112(%rsp), %rcx
	movq	48(%rsp), %rdx
	movq	56(%rsp), %r8
	movq	48(%rsp), %rax
	subq	%rax, %r8
	callq	memcpy
	movq	56(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_narrow@?$ctype@D@std@@MEBADDD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_narrow@?$ctype@D@std@@MEBADDD@Z"
	.globl	"?do_narrow@?$ctype@D@std@@MEBADDD@Z" # -- Begin function ?do_narrow@?$ctype@D@std@@MEBADDD@Z
	.p2align	4, 0x90
"?do_narrow@?$ctype@D@std@@MEBADDD@Z":  # @"?do_narrow@?$ctype@D@std@@MEBADDD@Z"
.seh_proc "?do_narrow@?$ctype@D@std@@MEBADDD@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movb	%r8b, 15(%rsp)
	movb	%dl, 14(%rsp)
	movq	%rcx, (%rsp)
	movb	14(%rsp), %al
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0facet@locale@std@@IEAA@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0facet@locale@std@@IEAA@_K@Z"
	.globl	"??0facet@locale@std@@IEAA@_K@Z" # -- Begin function ??0facet@locale@std@@IEAA@_K@Z
	.p2align	4, 0x90
"??0facet@locale@std@@IEAA@_K@Z":       # @"??0facet@locale@std@@IEAA@_K@Z"
.seh_proc "??0facet@locale@std@@IEAA@_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	callq	"??0_Facet_base@std@@QEAA@XZ"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7facet@locale@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	movq	48(%rsp), %rcx
                                        # kill: def $ecx killed $ecx killed $rcx
	movl	%ecx, 8(%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gctype_base@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gctype_base@std@@UEAAPEAXI@Z"
	.globl	"??_Gctype_base@std@@UEAAPEAXI@Z" # -- Begin function ??_Gctype_base@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gctype_base@std@@UEAAPEAXI@Z":      # @"??_Gctype_base@std@@UEAAPEAXI@Z"
.seh_proc "??_Gctype_base@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1ctype_base@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB78_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB78_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_Facet_base@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_Facet_base@std@@QEAA@XZ"
	.globl	"??0_Facet_base@std@@QEAA@XZ"   # -- Begin function ??0_Facet_base@std@@QEAA@XZ
	.p2align	4, 0x90
"??0_Facet_base@std@@QEAA@XZ":          # @"??0_Facet_base@std@@QEAA@XZ"
.seh_proc "??0_Facet_base@std@@QEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	leaq	"??_7_Facet_base@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gfacet@locale@std@@MEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gfacet@locale@std@@MEAAPEAXI@Z"
	.globl	"??_Gfacet@locale@std@@MEAAPEAXI@Z" # -- Begin function ??_Gfacet@locale@std@@MEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gfacet@locale@std@@MEAAPEAXI@Z":    # @"??_Gfacet@locale@std@@MEAAPEAXI@Z"
.seh_proc "??_Gfacet@locale@std@@MEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1facet@locale@std@@MEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB80_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB80_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_G_Facet_base@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G_Facet_base@std@@UEAAPEAXI@Z"
	.globl	"??_G_Facet_base@std@@UEAAPEAXI@Z" # -- Begin function ??_G_Facet_base@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_G_Facet_base@std@@UEAAPEAXI@Z":     # @"??_G_Facet_base@std@@UEAAPEAXI@Z"
.seh_proc "??_G_Facet_base@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movl	%edx, 12(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	%rax, 16(%rsp)
	ud2
	.seh_endproc
                                        # -- End function
	.def	"??1facet@locale@std@@MEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1facet@locale@std@@MEAA@XZ"
	.globl	"??1facet@locale@std@@MEAA@XZ"  # -- Begin function ??1facet@locale@std@@MEAA@XZ
	.p2align	4, 0x90
"??1facet@locale@std@@MEAA@XZ":         # @"??1facet@locale@std@@MEAA@XZ"
.seh_proc "??1facet@locale@std@@MEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1_Facet_base@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1_Facet_base@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_Facet_base@std@@UEAA@XZ"
	.globl	"??1_Facet_base@std@@UEAA@XZ"   # -- Begin function ??1_Facet_base@std@@UEAA@XZ
	.p2align	4, 0x90
"??1_Facet_base@std@@UEAA@XZ":          # @"??1_Facet_base@std@@UEAA@XZ"
.seh_proc "??1_Facet_base@std@@UEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ"
	.globl	"?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ" # -- Begin function ?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ
	.p2align	4, 0x90
"?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ": # @"?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ"
.seh_proc "?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movq	%rax, 56(%rsp)
	callq	_Getctype
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$ctype@D@std@@MEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$ctype@D@std@@MEAA@XZ"
	.globl	"??1?$ctype@D@std@@MEAA@XZ"     # -- Begin function ??1?$ctype@D@std@@MEAA@XZ
	.p2align	4, 0x90
"??1?$ctype@D@std@@MEAA@XZ":            # @"??1?$ctype@D@std@@MEAA@XZ"
.seh_proc "??1?$ctype@D@std@@MEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	leaq	"??_7?$ctype@D@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	callq	"?_Tidy@?$ctype@D@std@@IEAAXXZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"??1ctype_base@std@@UEAA@XZ"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Tidy@?$ctype@D@std@@IEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Tidy@?$ctype@D@std@@IEAAXXZ"
	.globl	"?_Tidy@?$ctype@D@std@@IEAAXXZ" # -- Begin function ?_Tidy@?$ctype@D@std@@IEAAXXZ
	.p2align	4, 0x90
"?_Tidy@?$ctype@D@std@@IEAAXXZ":        # @"?_Tidy@?$ctype@D@std@@IEAAXXZ"
.Lfunc_begin18:
.seh_proc "?_Tidy@?$ctype@D@std@@IEAAXXZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rcx
	movq	%rcx, -24(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	cmpl	32(%rcx), %eax
	jge	.LBB86_3
# %bb.1:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	24(%rax), %rcx
.Ltmp80:
	callq	free
.Ltmp81:
	jmp	.LBB86_2
.LBB86_2:
	jmp	.LBB86_8
.LBB86_3:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	cmpl	$0, 32(%rax)
	jge	.LBB86_7
# %bb.4:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	24(%rax), %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB86_6
# %bb.5:
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	callq	"??_V@YAXPEAX@Z"
.LBB86_6:
	jmp	.LBB86_7
.LBB86_7:
	jmp	.LBB86_8
.LBB86_8:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	40(%rax), %rcx
.Ltmp82:
	callq	free
.Ltmp83:
	jmp	.LBB86_9
.LBB86_9:
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Tidy@?$ctype@D@std@@IEAAXXZ")@IMGREL
	.section	.text,"xr",discard,"?_Tidy@?$ctype@D@std@@IEAAXXZ"
	.seh_endproc
	.def	"?dtor$10@?0??_Tidy@?$ctype@D@std@@IEAAXXZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0??_Tidy@?$ctype@D@std@@IEAAXXZ@4HA":
.seh_proc "?dtor$10@?0??_Tidy@?$ctype@D@std@@IEAAXXZ@4HA"
.LBB86_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end18:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Tidy@?$ctype@D@std@@IEAAXXZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Tidy@?$ctype@D@std@@IEAAXXZ"
	.p2align	2
"$cppxdata$?_Tidy@?$ctype@D@std@@IEAAXXZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?_Tidy@?$ctype@D@std@@IEAAXXZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?_Tidy@?$ctype@D@std@@IEAAXXZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Tidy@?$ctype@D@std@@IEAAXXZ":
	.long	-1                              # ToState
	.long	"?dtor$10@?0??_Tidy@?$ctype@D@std@@IEAAXXZ@4HA"@IMGREL # Action
"$ip2state$?_Tidy@?$ctype@D@std@@IEAAXXZ":
	.long	.Lfunc_begin18@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp80@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp83@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Tidy@?$ctype@D@std@@IEAAXXZ"
                                        # -- End function
	.def	"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z"
	.globl	"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z" # -- Begin function ??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z
	.p2align	4, 0x90
"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z": # @"??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z"
.seh_proc "??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z"
	.globl	"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z" # -- Begin function ??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z
	.p2align	4, 0x90
"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z": # @"??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z"
.seh_proc "??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0bad_cast@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0bad_cast@std@@QEAA@XZ"
	.globl	"??0bad_cast@std@@QEAA@XZ"      # -- Begin function ??0bad_cast@std@@QEAA@XZ
	.p2align	4, 0x90
"??0bad_cast@std@@QEAA@XZ":             # @"??0bad_cast@std@@QEAA@XZ"
.seh_proc "??0bad_cast@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	leaq	"??_C@_08EPJLHIJG@bad?5cast?$AA@"(%rip), %rdx
	movl	$1, %r8d
	callq	"??0exception@std@@QEAA@QEBDH@Z"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7bad_cast@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0bad_cast@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0bad_cast@std@@QEAA@AEBV01@@Z"
	.globl	"??0bad_cast@std@@QEAA@AEBV01@@Z" # -- Begin function ??0bad_cast@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0bad_cast@std@@QEAA@AEBV01@@Z":      # @"??0bad_cast@std@@QEAA@AEBV01@@Z"
.seh_proc "??0bad_cast@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0exception@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7bad_cast@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0exception@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0exception@std@@QEAA@AEBV01@@Z"
	.globl	"??0exception@std@@QEAA@AEBV01@@Z" # -- Begin function ??0exception@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0exception@std@@QEAA@AEBV01@@Z":     # @"??0exception@std@@QEAA@AEBV01@@Z"
.Lfunc_begin19:
.seh_proc "??0exception@std@@QEAA@AEBV01@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	leaq	"??_7exception@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	movq	%rax, %rdx
	addq	$8, %rdx
	xorps	%xmm0, %xmm0
	movups	%xmm0, 8(%rax)
	movq	-16(%rbp), %rcx
	addq	$8, %rcx
.Ltmp84:
	callq	__std_exception_copy
.Ltmp85:
	jmp	.LBB91_1
.LBB91_1:
	movq	-32(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0exception@std@@QEAA@AEBV01@@Z")@IMGREL
	.section	.text,"xr",discard,"??0exception@std@@QEAA@AEBV01@@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0exception@std@@QEAA@AEBV01@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0exception@std@@QEAA@AEBV01@@Z@4HA":
.seh_proc "?dtor$2@?0???0exception@std@@QEAA@AEBV01@@Z@4HA"
.LBB91_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end19:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0exception@std@@QEAA@AEBV01@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0exception@std@@QEAA@AEBV01@@Z"
	.p2align	2
"$cppxdata$??0exception@std@@QEAA@AEBV01@@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0exception@std@@QEAA@AEBV01@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0exception@std@@QEAA@AEBV01@@Z")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0exception@std@@QEAA@AEBV01@@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0exception@std@@QEAA@AEBV01@@Z@4HA"@IMGREL # Action
"$ip2state$??0exception@std@@QEAA@AEBV01@@Z":
	.long	.Lfunc_begin19@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp84@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp85@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0exception@std@@QEAA@AEBV01@@Z"
                                        # -- End function
	.def	"??1bad_cast@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1bad_cast@std@@UEAA@XZ"
	.globl	"??1bad_cast@std@@UEAA@XZ"      # -- Begin function ??1bad_cast@std@@UEAA@XZ
	.p2align	4, 0x90
"??1bad_cast@std@@UEAA@XZ":             # @"??1bad_cast@std@@UEAA@XZ"
.seh_proc "??1bad_cast@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1exception@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0exception@std@@QEAA@QEBDH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0exception@std@@QEAA@QEBDH@Z"
	.globl	"??0exception@std@@QEAA@QEBDH@Z" # -- Begin function ??0exception@std@@QEAA@QEBDH@Z
	.p2align	4, 0x90
"??0exception@std@@QEAA@QEBDH@Z":       # @"??0exception@std@@QEAA@QEBDH@Z"
.seh_proc "??0exception@std@@QEAA@QEBDH@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%r8d, 68(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	leaq	"??_7exception@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	addq	$8, %rcx
	xorl	%edx, %edx
	movl	$16, %r8d
	callq	memset
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	56(%rsp), %rcx
	movq	%rcx, 8(%rax)
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gbad_cast@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gbad_cast@std@@UEAAPEAXI@Z"
	.globl	"??_Gbad_cast@std@@UEAAPEAXI@Z" # -- Begin function ??_Gbad_cast@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gbad_cast@std@@UEAAPEAXI@Z":        # @"??_Gbad_cast@std@@UEAAPEAXI@Z"
.seh_proc "??_Gbad_cast@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1bad_cast@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB94_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB94_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?what@exception@std@@UEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?what@exception@std@@UEBAPEBDXZ"
	.globl	"?what@exception@std@@UEBAPEBDXZ" # -- Begin function ?what@exception@std@@UEBAPEBDXZ
	.p2align	4, 0x90
"?what@exception@std@@UEBAPEBDXZ":      # @"?what@exception@std@@UEBAPEBDXZ"
.seh_proc "?what@exception@std@@UEBAPEBDXZ"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	cmpq	$0, 8(%rax)
	je	.LBB95_2
# %bb.1:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	8(%rax), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	jmp	.LBB95_3
.LBB95_2:
	leaq	"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@"(%rip), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	jmp	.LBB95_3
.LBB95_3:
	movq	(%rsp), %rax                    # 8-byte Reload
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gexception@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gexception@std@@UEAAPEAXI@Z"
	.globl	"??_Gexception@std@@UEAAPEAXI@Z" # -- Begin function ??_Gexception@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gexception@std@@UEAAPEAXI@Z":       # @"??_Gexception@std@@UEAAPEAXI@Z"
.seh_proc "??_Gexception@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1exception@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB96_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB96_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1exception@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1exception@std@@UEAA@XZ"
	.globl	"??1exception@std@@UEAA@XZ"     # -- Begin function ??1exception@std@@UEAA@XZ
	.p2align	4, 0x90
"??1exception@std@@UEAA@XZ":            # @"??1exception@std@@UEAA@XZ"
.Lfunc_begin20:
.seh_proc "??1exception@std@@UEAA@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	48(%rsp), %rbp
	.seh_setframe %rbp, 48
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rcx
	leaq	"??_7exception@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	addq	$8, %rcx
.Ltmp86:
	callq	__std_exception_destroy
.Ltmp87:
	jmp	.LBB97_1
.LBB97_1:
	addq	$48, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??1exception@std@@UEAA@XZ")@IMGREL
	.section	.text,"xr",discard,"??1exception@std@@UEAA@XZ"
	.seh_endproc
	.def	"?dtor$2@?0???1exception@std@@UEAA@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???1exception@std@@UEAA@XZ@4HA":
.seh_proc "?dtor$2@?0???1exception@std@@UEAA@XZ@4HA"
.LBB97_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	48(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end20:
	.seh_handlerdata
	.section	.text,"xr",discard,"??1exception@std@@UEAA@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"??1exception@std@@UEAA@XZ"
	.p2align	2
"$cppxdata$??1exception@std@@UEAA@XZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??1exception@std@@UEAA@XZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??1exception@std@@UEAA@XZ")@IMGREL # IPToStateXData
	.long	40                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??1exception@std@@UEAA@XZ":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???1exception@std@@UEAA@XZ@4HA"@IMGREL # Action
"$ip2state$??1exception@std@@UEAA@XZ":
	.long	.Lfunc_begin20@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp86@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp87@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??1exception@std@@UEAA@XZ"
                                        # -- End function
	.def	"??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z"
	.globl	"??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z" # -- Begin function ??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z
	.p2align	4, 0x90
"??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z": # @"??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z"
.seh_proc "??$?0AEAPEAV_Facet_base@std@@@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@AEAPEAV_Facet_base@1@@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movb	%dl, 16(%rsp)
	movq	%r8, 8(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	8(%rsp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, (%rax)
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z"
	.globl	"??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z" # -- Begin function ??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z
	.p2align	4, 0x90
"??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z": # @"??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z"
.seh_proc "??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rdx, 16(%rsp)
	movq	%rcx, 8(%rsp)
	movq	8(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, (%rsp)
	movq	8(%rsp), %rax
	movq	$0, (%rax)
	movq	(%rsp), %rax
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ"
	.globl	"?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ" # -- Begin function ?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ
	.p2align	4, 0x90
"?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ": # @"?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ"
.seh_proc "?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z"
	.globl	"??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z" # -- Begin function ??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z
	.p2align	4, 0x90
"??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z": # @"??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z"
.seh_proc "??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	cmpq	$0, %rax
	je	.LBB101_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	(%rcx), %rax
	movl	$1, %edx
	callq	*(%rax)
.LBB101_2:
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	.globl	"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ" # -- Begin function ?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ
	.p2align	4, 0x90
"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ": # @"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
.seh_proc "?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	56(%rax), %rax
	cmpq	$0, (%rax)
	je	.LBB102_2
# %bb.1:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	80(%rax), %rax
	movl	(%rax), %eax
	movl	%eax, 4(%rsp)                   # 4-byte Spill
	jmp	.LBB102_3
.LBB102_2:
	xorl	%eax, %eax
	movl	%eax, 4(%rsp)                   # 4-byte Spill
	jmp	.LBB102_3
.LBB102_3:
	movl	4(%rsp), %eax                   # 4-byte Reload
	cltq
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	.globl	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z" # -- Begin function ?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z
	.p2align	4, 0x90
"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z": # @"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
.seh_proc "?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movzbl	(%rax), %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ"
	.globl	"?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ" # -- Begin function ?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ
	.p2align	4, 0x90
"?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ": # @"?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ"
.seh_proc "?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	56(%rax), %rax
	movq	(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	.globl	"?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ" # -- Begin function ?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ
	.p2align	4, 0x90
"?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ": # @"?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
.seh_proc "?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	80(%rax), %rcx
	movl	(%rcx), %edx
	addl	$-1, %edx
	movl	%edx, (%rcx)
	movq	56(%rax), %rcx
	movq	(%rcx), %rax
	addq	$1, %rax
	movq	%rax, (%rcx)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.globl	"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ" # -- Begin function ?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ
	.p2align	4, 0x90
"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ": # @"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
.seh_proc "?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	movq	%rax, %rcx
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	%rcx, %rax
	jge	.LBB106_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	movq	%rax, %rcx
	callq	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	movl	%eax, 36(%rsp)                  # 4-byte Spill
	jmp	.LBB106_3
.LBB106_2:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	(%rcx), %rax
	callq	*56(%rax)
	movl	%eax, 36(%rsp)                  # 4-byte Spill
.LBB106_3:
	movl	36(%rsp), %eax                  # 4-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	.globl	"?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ" # -- Begin function ?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ
	.p2align	4, 0x90
"?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ": # @"?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
.seh_proc "?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	80(%rax), %rcx
	movl	(%rcx), %edx
	addl	$-1, %edx
	movl	%edx, (%rcx)
	movq	56(%rax), %rcx
	movq	(%rcx), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, (%rcx)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.globl	"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z" # -- Begin function ?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z
	.p2align	4, 0x90
"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z": # @"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Lfunc_begin21:
.seh_proc "?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$176, %rsp
	.seh_stackalloc 176
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 40(%rbp)
	movq	%rdx, 32(%rbp)
	movq	%rcx, 24(%rbp)
	cmpq	$0, 24(%rbp)
	je	.LBB108_9
# %bb.1:
	movq	24(%rbp), %rax
	cmpq	$0, (%rax)
	jne	.LBB108_9
# %bb.2:
	movl	$16, %ecx
	callq	"??2@YAPEAX_K@Z"
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movb	$1, -81(%rbp)
	movq	32(%rbp), %rcx
	callq	"?_C_str@locale@std@@QEBAPEBDXZ"
	movq	%rax, %rdx
.Ltmp88:
	leaq	-80(%rbp), %rcx
	callq	"??0_Locinfo@std@@QEAA@PEBD@Z"
.Ltmp89:
	jmp	.LBB108_3
.LBB108_3:
.Ltmp90:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	xorl	%eax, %eax
	movl	%eax, %r8d
	leaq	-80(%rbp), %rdx
	callq	"??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
.Ltmp91:
	jmp	.LBB108_4
.LBB108_4:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	movb	$0, -81(%rbp)
	movq	24(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	jmp	.LBB108_9
.LBB108_9:
	movl	$4, %eax
	addq	$176, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$5@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$5@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$5@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB108_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$6@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$6@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB108_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	testb	$1, -81(%rbp)
	jne	.LBB108_7
	jmp	.LBB108_8
.LBB108_7:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB108_8:
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end21:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.p2align	2
"$cppxdata$?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # IPToStateXData
	.long	168                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	-1                              # ToState
	.long	"?dtor$6@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$5@?0??_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
"$ip2state$?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	.Lfunc_begin21@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp88@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp90@IMGREL+1                # IP
	.long	1                               # ToState
	.long	.Ltmp91@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
                                        # -- End function
	.def	"??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.globl	"??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z" # -- Begin function ??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z
	.p2align	4, 0x90
"??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z": # @"??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
.seh_proc "??0?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r8, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rdx
	callq	"??0facet@locale@std@@IEAA@_K@Z"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leaq	"??_7?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	movq	56(%rsp), %rdx
	callq	"?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
	.globl	"?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z" # -- Begin function ?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z
	.p2align	4, 0x90
"?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z": # @"?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
.seh_proc "?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
	.globl	"??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z" # -- Begin function ??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z
	.p2align	4, 0x90
"??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z": # @"??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
.seh_proc "??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB111_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB111_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
.Lfunc_begin22:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$208, %rsp
	.seh_stackalloc 208
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 72(%rbp)
	movq	%r9, -40(%rbp)                  # 8-byte Spill
	movq	%r8, -48(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -80(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -72(%rbp)                  # 8-byte Spill
	movq	144(%rbp), %r9
	movq	136(%rbp), %r9
	movq	128(%rbp), %r9
	movq	%r8, 64(%rbp)
	movq	%rax, 56(%rbp)
	movq	56(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	128(%rbp), %rcx
	leaq	-8(%rbp), %rdx
	movq	%rdx, -64(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-64(%rbp), %rdx                 # 8-byte Reload
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %r8                  # 8-byte Reload
	movq	-40(%rbp), %r9                  # 8-byte Reload
.Ltmp92:
	movq	%rsp, %rax
	movq	%rdx, 40(%rax)
	movl	$2048, 32(%rax)                 # imm = 0x800
	leaq	16(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp93:
	movl	%eax, -28(%rbp)                 # 4-byte Spill
	jmp	.LBB112_1
.LBB112_1:
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-28(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 12(%rbp)
	movsbl	16(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB112_4
# %bb.2:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
	movq	144(%rbp), %rax
	movq	$0, (%rax)
	jmp	.LBB112_8
.LBB112_4:
	movl	12(%rbp), %r8d
	leaq	16(%rbp), %rcx
	leaq	-24(%rbp), %rdx
	leaq	-12(%rbp), %r9
	callq	_Stoullx
	movq	%rax, %rcx
	movq	144(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	16(%rbp), %rax
	cmpq	%rax, -24(%rbp)
	je	.LBB112_6
# %bb.5:
	cmpl	$0, -12(%rbp)
	je	.LBB112_7
.LBB112_6:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
	movq	144(%rbp), %rax
	movq	$0, (%rax)
.LBB112_7:
	jmp	.LBB112_8
.LBB112_8:
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB112_9
	jmp	.LBB112_10
.LBB112_9:
	movq	136(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB112_10:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$208, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
	.seh_endproc
	.def	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z@4HA":
.seh_proc "?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z@4HA"
.LBB112_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end22:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z")@IMGREL # IPToStateXData
	.long	200                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z":
	.long	.Lfunc_begin22@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp92@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp93@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z"
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z"
# %bb.0:
	subq	$152, %rsp
	.seh_stackalloc 152
	.seh_endprologue
	movq	%r8, 56(%rsp)                   # 8-byte Spill
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	208(%rsp), %rax
	movq	200(%rsp), %rax
	movq	192(%rsp), %rax
	movq	%rdx, 144(%rsp)
	movq	%rcx, 136(%rsp)
	movq	136(%rsp), %rcx
	movq	200(%rsp), %r10
	movq	192(%rsp), %r11
	movq	(%r9), %rax
	movq	%rax, 96(%rsp)
	movq	8(%r9), %rax
	movq	%rax, 104(%rsp)
	movq	(%r8), %rax
	movq	%rax, 80(%rsp)
	movq	8(%r8), %rax
	movq	%rax, 88(%rsp)
	leaq	112(%rsp), %rdx
	leaq	80(%rsp), %r8
	leaq	96(%rsp), %r9
	leaq	128(%rsp), %rax
	movq	%r11, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z"
	movq	56(%rsp), %r8                   # 8-byte Reload
	movq	64(%rsp), %rdx                  # 8-byte Reload
	movq	72(%rsp), %rax                  # 8-byte Reload
	movq	112(%rsp), %rcx
	movq	%rcx, (%r8)
	movq	120(%rsp), %rcx
	movq	%rcx, 8(%r8)
	movsd	128(%rsp), %xmm0                # xmm0 = mem[0],zero
	movq	208(%rsp), %rcx
	movsd	%xmm0, (%rcx)
	movq	(%r8), %rcx
	movq	%rcx, (%rdx)
	movq	8(%r8), %rcx
	movq	%rcx, 8(%rdx)
	addq	$152, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z"
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z"
# %bb.0:
	subq	$920, %rsp                      # imm = 0x398
	.seh_stackalloc 920
	.seh_endprologue
	movq	%r9, 80(%rsp)                   # 8-byte Spill
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	movq	%rdx, %r8
	movq	80(%rsp), %rdx                  # 8-byte Reload
	movq	%r8, 48(%rsp)                   # 8-byte Spill
	movq	%rcx, %rax
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, 56(%rsp)                   # 8-byte Spill
	movq	976(%rsp), %r9
	movq	968(%rsp), %r9
	movq	960(%rsp), %r9
	movq	%r8, 912(%rsp)
	movq	%rax, 904(%rsp)
	movq	904(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	64(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %r8                   # 8-byte Reload
	movq	80(%rsp), %r9                   # 8-byte Reload
	movl	$1000000000, 108(%rsp)          # imm = 0x3B9ACA00
	movq	960(%rsp), %r10
	leaq	112(%rsp), %rdx
	leaq	108(%rsp), %rax
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	movl	%eax, 104(%rsp)
	movsbl	112(%rsp), %eax
	cmpl	$0, %eax
	jne	.LBB114_2
# %bb.1:
	movq	968(%rsp), %rax
	movl	$2, (%rax)
	movq	976(%rsp), %rax
	xorps	%xmm0, %xmm0
	movsd	%xmm0, (%rax)
	jmp	.LBB114_10
.LBB114_2:
	movl	104(%rsp), %r8d
	leaq	112(%rsp), %rcx
	leaq	88(%rsp), %rdx
	leaq	100(%rsp), %r9
	callq	"?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z"
	movq	976(%rsp), %rax
	movsd	%xmm0, (%rax)
	movq	88(%rsp), %rax
	leaq	112(%rsp), %rcx
	cmpq	%rcx, %rax
	je	.LBB114_4
# %bb.3:
	cmpl	$0, 100(%rsp)
	je	.LBB114_5
.LBB114_4:
	movq	968(%rsp), %rax
	movl	$2, (%rax)
	movq	976(%rsp), %rax
	xorps	%xmm0, %xmm0
	movsd	%xmm0, (%rax)
	jmp	.LBB114_9
.LBB114_5:
	cmpl	$1000000000, 108(%rsp)          # imm = 0x3B9ACA00
	je	.LBB114_8
# %bb.6:
	cmpl	$0, 108(%rsp)
	je	.LBB114_8
# %bb.7:
	movl	108(%rsp), %edx
	shll	$2, %edx
	movq	976(%rsp), %rax
	movsd	(%rax), %xmm0                   # xmm0 = mem[0],zero
	callq	ldexp
	movq	976(%rsp), %rax
	movsd	%xmm0, (%rax)
.LBB114_8:
	jmp	.LBB114_9
.LBB114_9:
	jmp	.LBB114_10
.LBB114_10:
	movq	80(%rsp), %rdx                  # 8-byte Reload
	movq	72(%rsp), %rcx                  # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB114_11
	jmp	.LBB114_12
.LBB114_11:
	movq	968(%rsp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB114_12:
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %rdx                  # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$920, %rsp                      # imm = 0x398
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z"
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z"
# %bb.0:
	subq	$920, %rsp                      # imm = 0x398
	.seh_stackalloc 920
	.seh_endprologue
	movq	%r9, 80(%rsp)                   # 8-byte Spill
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	movq	%rdx, %r8
	movq	80(%rsp), %rdx                  # 8-byte Reload
	movq	%r8, 48(%rsp)                   # 8-byte Spill
	movq	%rcx, %rax
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, 56(%rsp)                   # 8-byte Spill
	movq	976(%rsp), %r9
	movq	968(%rsp), %r9
	movq	960(%rsp), %r9
	movq	%r8, 912(%rsp)
	movq	%rax, 904(%rsp)
	movq	904(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	64(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %r8                   # 8-byte Reload
	movq	80(%rsp), %r9                   # 8-byte Reload
	movl	$1000000000, 108(%rsp)          # imm = 0x3B9ACA00
	movq	960(%rsp), %r10
	leaq	112(%rsp), %rdx
	leaq	108(%rsp), %rax
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	movl	%eax, 104(%rsp)
	movsbl	112(%rsp), %eax
	cmpl	$0, %eax
	jne	.LBB115_2
# %bb.1:
	movq	968(%rsp), %rax
	movl	$2, (%rax)
	movq	976(%rsp), %rax
	xorps	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	jmp	.LBB115_10
.LBB115_2:
	movl	104(%rsp), %r8d
	leaq	112(%rsp), %rcx
	leaq	88(%rsp), %rdx
	leaq	100(%rsp), %r9
	callq	"?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z"
	movq	976(%rsp), %rax
	movss	%xmm0, (%rax)
	movq	88(%rsp), %rax
	leaq	112(%rsp), %rcx
	cmpq	%rcx, %rax
	je	.LBB115_4
# %bb.3:
	cmpl	$0, 100(%rsp)
	je	.LBB115_5
.LBB115_4:
	movq	968(%rsp), %rax
	movl	$2, (%rax)
	movq	976(%rsp), %rax
	xorps	%xmm0, %xmm0
	movss	%xmm0, (%rax)
	jmp	.LBB115_9
.LBB115_5:
	cmpl	$1000000000, 108(%rsp)          # imm = 0x3B9ACA00
	je	.LBB115_8
# %bb.6:
	cmpl	$0, 108(%rsp)
	je	.LBB115_8
# %bb.7:
	movl	108(%rsp), %edx
	shll	$2, %edx
	movq	976(%rsp), %rax
	movss	(%rax), %xmm0                   # xmm0 = mem[0],zero,zero,zero
	callq	ldexpf
	movq	976(%rsp), %rax
	movss	%xmm0, (%rax)
.LBB115_8:
	jmp	.LBB115_9
.LBB115_9:
	jmp	.LBB115_10
.LBB115_10:
	movq	80(%rsp), %rdx                  # 8-byte Reload
	movq	72(%rsp), %rcx                  # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB115_11
	jmp	.LBB115_12
.LBB115_11:
	movq	968(%rsp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB115_12:
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %rdx                  # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$920, %rsp                      # imm = 0x398
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
.Lfunc_begin23:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$208, %rsp
	.seh_stackalloc 208
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 72(%rbp)
	movq	%r9, -40(%rbp)                  # 8-byte Spill
	movq	%r8, -48(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -80(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -72(%rbp)                  # 8-byte Spill
	movq	144(%rbp), %r9
	movq	136(%rbp), %r9
	movq	128(%rbp), %r9
	movq	%r8, 64(%rbp)
	movq	%rax, 56(%rbp)
	movq	56(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	128(%rbp), %rcx
	leaq	-8(%rbp), %rdx
	movq	%rdx, -64(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	128(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	-64(%rbp), %r10                 # 8-byte Reload
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %r8                  # 8-byte Reload
	movq	-40(%rbp), %r9                  # 8-byte Reload
	movl	%eax, %edx
.Ltmp94:
	movq	%rsp, %rax
	movq	%r10, 40(%rax)
	movl	%edx, 32(%rax)
	leaq	16(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp95:
	movl	%eax, -28(%rbp)                 # 4-byte Spill
	jmp	.LBB116_1
.LBB116_1:
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-28(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 12(%rbp)
	movsbl	16(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB116_4
# %bb.2:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
	movq	144(%rbp), %rax
	movq	$0, (%rax)
	jmp	.LBB116_8
.LBB116_4:
	movl	12(%rbp), %r8d
	leaq	16(%rbp), %rcx
	leaq	-24(%rbp), %rdx
	leaq	-12(%rbp), %r9
	callq	_Stoullx
	movq	%rax, %rcx
	movq	144(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	16(%rbp), %rax
	cmpq	%rax, -24(%rbp)
	je	.LBB116_6
# %bb.5:
	cmpl	$0, -12(%rbp)
	je	.LBB116_7
.LBB116_6:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
.LBB116_7:
	jmp	.LBB116_8
.LBB116_8:
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB116_9
	jmp	.LBB116_10
.LBB116_9:
	movq	136(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB116_10:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$208, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
	.seh_endproc
	.def	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z@4HA":
.seh_proc "?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z@4HA"
.LBB116_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end23:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z")@IMGREL # IPToStateXData
	.long	200                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z":
	.long	.Lfunc_begin23@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp94@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp95@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
.Lfunc_begin24:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$208, %rsp
	.seh_stackalloc 208
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 72(%rbp)
	movq	%r9, -32(%rbp)                  # 8-byte Spill
	movq	%r8, -40(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -72(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -64(%rbp)                  # 8-byte Spill
	movq	144(%rbp), %r9
	movq	136(%rbp), %r9
	movq	128(%rbp), %r9
	movq	%r8, 64(%rbp)
	movq	%rax, 56(%rbp)
	movq	56(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	128(%rbp), %rcx
	leaq	-8(%rbp), %rdx
	movq	%rdx, -56(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	128(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	-56(%rbp), %r10                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %r8                  # 8-byte Reload
	movq	-32(%rbp), %r9                  # 8-byte Reload
	movl	%eax, %edx
.Ltmp96:
	movq	%rsp, %rax
	movq	%r10, 40(%rax)
	movl	%edx, 32(%rax)
	leaq	16(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp97:
	movl	%eax, -24(%rbp)                 # 4-byte Spill
	jmp	.LBB117_1
.LBB117_1:
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-24(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 12(%rbp)
	movsbl	16(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB117_4
# %bb.2:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
	movq	144(%rbp), %rax
	movq	$0, (%rax)
	jmp	.LBB117_8
.LBB117_4:
	movl	12(%rbp), %r8d
	leaq	16(%rbp), %rcx
	leaq	-16(%rbp), %rdx
	leaq	-20(%rbp), %r9
	callq	_Stollx
	movq	%rax, %rcx
	movq	144(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	16(%rbp), %rax
	cmpq	%rax, -16(%rbp)
	je	.LBB117_6
# %bb.5:
	cmpl	$0, -20(%rbp)
	je	.LBB117_7
.LBB117_6:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
.LBB117_7:
	jmp	.LBB117_8
.LBB117_8:
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB117_9
	jmp	.LBB117_10
.LBB117_9:
	movq	136(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB117_10:
	movq	-64(%rbp), %rax                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$208, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
	.seh_endproc
	.def	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z@4HA":
.seh_proc "?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z@4HA"
.LBB117_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end24:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z")@IMGREL # IPToStateXData
	.long	200                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z":
	.long	.Lfunc_begin24@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp96@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp97@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
.Lfunc_begin25:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$208, %rsp
	.seh_stackalloc 208
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 72(%rbp)
	movq	%r9, -32(%rbp)                  # 8-byte Spill
	movq	%r8, -40(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -72(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -64(%rbp)                  # 8-byte Spill
	movq	144(%rbp), %r9
	movq	136(%rbp), %r9
	movq	128(%rbp), %r9
	movq	%r8, 64(%rbp)
	movq	%rax, 56(%rbp)
	movq	56(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	128(%rbp), %rcx
	leaq	-8(%rbp), %rdx
	movq	%rdx, -56(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	128(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	-56(%rbp), %r10                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %r8                  # 8-byte Reload
	movq	-32(%rbp), %r9                  # 8-byte Reload
	movl	%eax, %edx
.Ltmp98:
	movq	%rsp, %rax
	movq	%r10, 40(%rax)
	movl	%edx, 32(%rax)
	leaq	16(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp99:
	movl	%eax, -24(%rbp)                 # 4-byte Spill
	jmp	.LBB118_1
.LBB118_1:
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-24(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 12(%rbp)
	movsbl	16(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB118_4
# %bb.2:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
	movq	144(%rbp), %rax
	movl	$0, (%rax)
	jmp	.LBB118_8
.LBB118_4:
	movl	12(%rbp), %r8d
	leaq	16(%rbp), %rcx
	leaq	-16(%rbp), %rdx
	leaq	-20(%rbp), %r9
	callq	_Stoulx
	movl	%eax, %ecx
	movq	144(%rbp), %rax
	movl	%ecx, (%rax)
	leaq	16(%rbp), %rax
	cmpq	%rax, -16(%rbp)
	je	.LBB118_6
# %bb.5:
	cmpl	$0, -20(%rbp)
	je	.LBB118_7
.LBB118_6:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
.LBB118_7:
	jmp	.LBB118_8
.LBB118_8:
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB118_9
	jmp	.LBB118_10
.LBB118_9:
	movq	136(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB118_10:
	movq	-64(%rbp), %rax                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$208, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	.seh_endproc
	.def	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z@4HA":
.seh_proc "?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z@4HA"
.LBB118_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end25:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z")@IMGREL # IPToStateXData
	.long	200                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z":
	.long	.Lfunc_begin25@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp98@IMGREL+1                # IP
	.long	0                               # ToState
	.long	.Ltmp99@IMGREL+1                # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
.Lfunc_begin26:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$208, %rsp
	.seh_stackalloc 208
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 72(%rbp)
	movq	%r9, -32(%rbp)                  # 8-byte Spill
	movq	%r8, -40(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -72(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -64(%rbp)                  # 8-byte Spill
	movq	144(%rbp), %r9
	movq	136(%rbp), %r9
	movq	128(%rbp), %r9
	movq	%r8, 64(%rbp)
	movq	%rax, 56(%rbp)
	movq	56(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	128(%rbp), %rcx
	leaq	-8(%rbp), %rdx
	movq	%rdx, -56(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	128(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	-56(%rbp), %r10                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %r8                  # 8-byte Reload
	movq	-32(%rbp), %r9                  # 8-byte Reload
	movl	%eax, %edx
.Ltmp100:
	movq	%rsp, %rax
	movq	%r10, 40(%rax)
	movl	%edx, 32(%rax)
	leaq	16(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp101:
	movl	%eax, -24(%rbp)                 # 4-byte Spill
	jmp	.LBB119_1
.LBB119_1:
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-24(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 12(%rbp)
	movsbl	16(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB119_4
# %bb.2:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
	movq	144(%rbp), %rax
	movl	$0, (%rax)
	jmp	.LBB119_8
.LBB119_4:
	movl	12(%rbp), %r8d
	leaq	16(%rbp), %rcx
	leaq	-16(%rbp), %rdx
	leaq	-20(%rbp), %r9
	callq	_Stolx
	movl	%eax, %ecx
	movq	144(%rbp), %rax
	movl	%ecx, (%rax)
	leaq	16(%rbp), %rax
	cmpq	%rax, -16(%rbp)
	je	.LBB119_6
# %bb.5:
	cmpl	$0, -20(%rbp)
	je	.LBB119_7
.LBB119_6:
	movq	136(%rbp), %rax
	movl	$2, (%rax)
.LBB119_7:
	jmp	.LBB119_8
.LBB119_8:
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB119_9
	jmp	.LBB119_10
.LBB119_9:
	movq	136(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB119_10:
	movq	-64(%rbp), %rax                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$208, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.seh_endproc
	.def	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z@4HA":
.seh_proc "?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z@4HA"
.LBB119_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end26:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z")@IMGREL # IPToStateXData
	.long	200                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z":
	.long	.Lfunc_begin26@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp100@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp101@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z"
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z"
# %bb.0:
	subq	$152, %rsp
	.seh_stackalloc 152
	.seh_endprologue
	movq	%r8, 56(%rsp)                   # 8-byte Spill
	movq	%rdx, 64(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	208(%rsp), %rax
	movq	200(%rsp), %rax
	movq	192(%rsp), %rax
	movq	%rdx, 144(%rsp)
	movq	%rcx, 136(%rsp)
	movq	136(%rsp), %rcx
	movq	200(%rsp), %r10
	movq	192(%rsp), %r11
	movq	(%r9), %rax
	movq	%rax, 96(%rsp)
	movq	8(%r9), %rax
	movq	%rax, 104(%rsp)
	movq	(%r8), %rax
	movq	%rax, 80(%rsp)
	movq	8(%r8), %rax
	movq	%rax, 88(%rsp)
	leaq	112(%rsp), %rdx
	leaq	80(%rsp), %r8
	leaq	96(%rsp), %r9
	leaq	132(%rsp), %rax
	movq	%r11, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	movq	56(%rsp), %r8                   # 8-byte Reload
	movq	64(%rsp), %rdx                  # 8-byte Reload
	movq	72(%rsp), %rax                  # 8-byte Reload
	movq	112(%rsp), %rcx
	movq	%rcx, (%r8)
	movq	120(%rsp), %rcx
	movq	%rcx, 8(%r8)
	movl	132(%rsp), %r9d
	movq	208(%rsp), %rcx
	movl	%r9d, (%rcx)
	movq	(%r8), %rcx
	movq	%rcx, (%rdx)
	movq	8(%r8), %rcx
	movq	%rcx, 8(%rdx)
	addq	$152, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
.Lfunc_begin27:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$224, %rsp
	.seh_stackalloc 224
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 88(%rbp)
	movq	%r9, -40(%rbp)                  # 8-byte Spill
	movq	%r8, -48(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -80(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -72(%rbp)                  # 8-byte Spill
	movq	160(%rbp), %r9
	movq	152(%rbp), %r9
	movq	144(%rbp), %r9
	movq	%r8, 80(%rbp)
	movq	%rax, 72(%rbp)
	movq	72(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	144(%rbp), %rcx
	leaq	8(%rbp), %rdx
	movq	%rdx, -64(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	144(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	-64(%rbp), %r10                 # 8-byte Reload
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %r8                  # 8-byte Reload
	movq	-40(%rbp), %r9                  # 8-byte Reload
	movl	%eax, %edx
.Ltmp102:
	movq	%rsp, %rax
	movq	%r10, 40(%rax)
	movl	%edx, 32(%rax)
	leaq	32(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp103:
	movl	%eax, -28(%rbp)                 # 4-byte Spill
	jmp	.LBB121_1
.LBB121_1:
	leaq	8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-28(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 28(%rbp)
	movsbl	32(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB121_4
# %bb.2:
	movq	152(%rbp), %rax
	movl	$2, (%rax)
	movq	160(%rbp), %rax
	movw	$0, (%rax)
	jmp	.LBB121_14
.LBB121_4:
	movsbl	32(%rbp), %eax
	cmpl	$45, %eax
	sete	%al
	andb	$1, %al
	movb	%al, 7(%rbp)
	leaq	32(%rbp), %rax
	movq	%rax, -8(%rbp)
	testb	$1, 7(%rbp)
	je	.LBB121_6
# %bb.5:
	movq	-8(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -8(%rbp)
.LBB121_6:
	movl	28(%rbp), %r8d
	movq	-8(%rbp), %rcx
	leaq	-16(%rbp), %rdx
	leaq	-20(%rbp), %r9
	callq	_Stoulx
	movl	%eax, -24(%rbp)
	movl	-24(%rbp), %eax
	movw	%ax, %cx
	movq	160(%rbp), %rax
	movw	%cx, (%rax)
	movq	-16(%rbp), %rax
	cmpq	-8(%rbp), %rax
	je	.LBB121_9
# %bb.7:
	cmpl	$0, -20(%rbp)
	jne	.LBB121_9
# %bb.8:
	cmpl	$65535, -24(%rbp)               # imm = 0xFFFF
	jbe	.LBB121_10
.LBB121_9:
	movq	152(%rbp), %rax
	movl	$2, (%rax)
	movq	160(%rbp), %rax
	movw	$-1, (%rax)
	jmp	.LBB121_13
.LBB121_10:
	testb	$1, 7(%rbp)
	je	.LBB121_12
# %bb.11:
	movq	160(%rbp), %rax
	movzwl	(%rax), %ecx
	xorl	%eax, %eax
	subl	%ecx, %eax
	movw	%ax, %cx
	movq	160(%rbp), %rax
	movw	%cx, (%rax)
.LBB121_12:
	jmp	.LBB121_13
.LBB121_13:
	jmp	.LBB121_14
.LBB121_14:
	movq	-40(%rbp), %rdx                 # 8-byte Reload
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB121_15
	jmp	.LBB121_16
.LBB121_15:
	movq	152(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB121_16:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$224, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
	.seh_endproc
	.def	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z@4HA":
.seh_proc "?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z@4HA"
.LBB121_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end27:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z")@IMGREL # IPToStateXData
	.long	216                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z":
	.long	.Lfunc_begin27@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp102@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp103@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
                                        # -- End function
	.def	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.globl	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z" # -- Begin function ?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z
	.p2align	4, 0x90
"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z": # @"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
.Lfunc_begin28:
.seh_proc "?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$336, %rsp                      # imm = 0x150
	.seh_stackalloc 336
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 200(%rbp)
	movq	%r9, -48(%rbp)                  # 8-byte Spill
	movq	%r8, -32(%rbp)                  # 8-byte Spill
	movq	%rdx, %r8
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	%r8, -40(%rbp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	%r8, %r9
	movq	%r9, -24(%rbp)                  # 8-byte Spill
	movq	272(%rbp), %r9
	movq	264(%rbp), %r9
	movq	256(%rbp), %r9
	movq	%r8, 192(%rbp)
	movq	%rax, 184(%rbp)
	movq	184(%rbp), %rax
	movq	%rax, -16(%rbp)                 # 8-byte Spill
	callq	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	movq	256(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$16384, %eax                    # imm = 0x4000
	cmpl	$0, %eax
	je	.LBB122_17
# %bb.1:
	movq	256(%rbp), %rcx
	leaq	160(%rbp), %rdx
	movq	%rdx, -64(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-64(%rbp), %rcx                 # 8-byte Reload
.Ltmp106:
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Ltmp107:
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	jmp	.LBB122_2
.LBB122_2:
	leaq	160(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 176(%rbp)
	xorl	%eax, %eax
	movb	%al, %r8b
	leaq	128(%rbp), %rcx
	movl	$1, %edx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	movq	176(%rbp), %rcx
.Ltmp108:
	leaq	96(%rbp), %rdx
	callq	"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp109:
	jmp	.LBB122_3
.LBB122_3:
.Ltmp110:
	leaq	128(%rbp), %rcx
	leaq	96(%rbp), %rdx
	callq	"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z"
.Ltmp111:
	jmp	.LBB122_4
.LBB122_4:
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
.Ltmp112:
	xorl	%eax, %eax
	movb	%al, %dl
	leaq	128(%rbp), %rcx
	callq	"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
.Ltmp113:
	jmp	.LBB122_5
.LBB122_5:
	movq	176(%rbp), %rcx
.Ltmp114:
	leaq	64(%rbp), %rdx
	callq	"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp115:
	jmp	.LBB122_6
.LBB122_6:
.Ltmp116:
	leaq	128(%rbp), %rcx
	leaq	64(%rbp), %rdx
	callq	"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z"
.Ltmp117:
	jmp	.LBB122_7
.LBB122_7:
	leaq	64(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	leaq	128(%rbp), %rcx
	callq	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	%rax, %r9
.Ltmp118:
	movq	%rsp, %rax
	movb	$1, 32(%rax)
	movl	$2, %r8d
	callq	"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
.Ltmp119:
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	jmp	.LBB122_8
.LBB122_8:
	movl	-68(%rbp), %eax                 # 4-byte Reload
	testl	%eax, %eax
	je	.LBB122_12
	jmp	.LBB122_33
.LBB122_33:
	movl	-68(%rbp), %eax                 # 4-byte Reload
	subl	$1, %eax
	je	.LBB122_13
	jmp	.LBB122_14
.LBB122_12:
	movq	272(%rbp), %rax
	movb	$0, (%rax)
	jmp	.LBB122_15
.LBB122_13:
	movq	272(%rbp), %rax
	movb	$1, (%rax)
	jmp	.LBB122_15
.LBB122_14:
	movq	272(%rbp), %rax
	movb	$0, (%rax)
	movq	264(%rbp), %rax
	movl	$2, (%rax)
.LBB122_15:
	leaq	128(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	jmp	.LBB122_30
.LBB122_17:
	movq	256(%rbp), %rcx
	leaq	8(%rbp), %rdx
	movq	%rdx, -80(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	256(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	-80(%rbp), %r10                 # 8-byte Reload
	movq	-16(%rbp), %rcx                 # 8-byte Reload
	movq	-32(%rbp), %r8                  # 8-byte Reload
	movq	-48(%rbp), %r9                  # 8-byte Reload
	movl	%eax, %edx
.Ltmp104:
	movq	%rsp, %rax
	movq	%r10, 40(%rax)
	movl	%edx, 32(%rax)
	leaq	32(%rbp), %rdx
	callq	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Ltmp105:
	movl	%eax, -72(%rbp)                 # 4-byte Spill
	jmp	.LBB122_18
.LBB122_18:
	leaq	8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movl	-72(%rbp), %eax                 # 4-byte Reload
	movl	%eax, 28(%rbp)
	movsbl	32(%rbp), %eax
	cmpl	$0, %eax
	jne	.LBB122_21
# %bb.19:
	movq	272(%rbp), %rax
	movb	$0, (%rax)
	movq	264(%rbp), %rax
	movl	$2, (%rax)
	jmp	.LBB122_29
.LBB122_21:
	movl	28(%rbp), %r8d
	leaq	32(%rbp), %rcx
	movq	%rbp, %rdx
	leaq	-4(%rbp), %r9
	callq	_Stolx
	movl	%eax, -8(%rbp)
	leaq	32(%rbp), %rax
	cmpq	%rax, (%rbp)
	je	.LBB122_23
# %bb.22:
	cmpl	$0, -4(%rbp)
	je	.LBB122_24
.LBB122_23:
	movq	272(%rbp), %rax
	movb	$1, (%rax)
	movq	264(%rbp), %rax
	movl	$2, (%rax)
	jmp	.LBB122_28
.LBB122_24:
	cmpl	$0, -8(%rbp)
	setne	%cl
	movq	272(%rbp), %rax
	andb	$1, %cl
	movb	%cl, (%rax)
	cmpl	$0, -8(%rbp)
	je	.LBB122_27
# %bb.25:
	cmpl	$1, -8(%rbp)
	je	.LBB122_27
# %bb.26:
	movq	264(%rbp), %rax
	movl	$2, (%rax)
.LBB122_27:
	jmp	.LBB122_28
.LBB122_28:
	jmp	.LBB122_29
.LBB122_29:
	jmp	.LBB122_30
.LBB122_30:
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	testb	$1, %al
	jne	.LBB122_31
	jmp	.LBB122_32
.LBB122_31:
	movq	264(%rbp), %rax
	movl	(%rax), %ecx
	orl	$1, %ecx
	movl	%ecx, (%rax)
.LBB122_32:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$336, %rsp                      # imm = 0x150
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z")@IMGREL
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_endproc
	.def	"?dtor$9@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$9@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA":
.seh_proc "?dtor$9@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"
.LBB122_9:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	160(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_endproc
	.def	"?dtor$10@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA":
.seh_proc "?dtor$10@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"
.LBB122_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_endproc
	.def	"?dtor$11@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$11@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA":
.seh_proc "?dtor$11@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"
.LBB122_11:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	64(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_endproc
	.def	"?dtor$16@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$16@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA":
.seh_proc "?dtor$16@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"
.LBB122_16:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	128(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_endproc
	.def	"?dtor$20@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$20@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA":
.seh_proc "?dtor$20@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"
.LBB122_20:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	8(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end28:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
	.p2align	2
"$cppxdata$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z":
	.long	429065506                       # MagicNumber
	.long	5                               # MaxState
	.long	("$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	10                              # IPMapEntries
	.long	("$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z")@IMGREL # IPToStateXData
	.long	328                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z":
	.long	-1                              # ToState
	.long	"?dtor$9@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$16@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$11@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$10@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$20@?0??do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z@4HA"@IMGREL # Action
"$ip2state$?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z":
	.long	.Lfunc_begin28@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp106@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp107@IMGREL+1               # IP
	.long	-1                              # ToState
	.long	.Ltmp108@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp110@IMGREL+1               # IP
	.long	3                               # ToState
	.long	.Ltmp112@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp116@IMGREL+1               # IP
	.long	2                               # ToState
	.long	.Ltmp118@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp104@IMGREL+1               # IP
	.long	4                               # ToState
	.long	.Ltmp105@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"
                                        # -- End function
	.def	"??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
	.globl	"??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ" # -- Begin function ??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ
	.p2align	4, 0x90
"??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ": # @"??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
.seh_proc "??1?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1facet@locale@std@@MEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	.globl	"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z" # -- Begin function ??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z
	.p2align	4, 0x90
"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z": # @"??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.seh_proc "??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.globl	"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z" # -- Begin function ?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z
	.p2align	4, 0x90
"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z": # @"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
.Lfunc_begin29:
.seh_proc "?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$336, %rsp                      # imm = 0x150
	.seh_stackalloc 336
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 200(%rbp)
	movq	264(%rbp), %rax
	movl	256(%rbp), %eax
	movq	%r9, 192(%rbp)
	movq	%r8, 184(%rbp)
	movq	%rdx, 176(%rbp)
	movq	%rcx, 168(%rbp)
	movq	264(%rbp), %rcx
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	movq	%rax, 160(%rbp)
	movq	160(%rbp), %rcx
	leaq	128(%rbp), %rdx
	callq	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	leaq	128(%rbp), %rcx
	callq	"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB125_1
	jmp	.LBB125_2
.LBB125_1:
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	movb	%al, -17(%rbp)                  # 1-byte Spill
	jmp	.LBB125_4
.LBB125_2:
	movq	160(%rbp), %rcx
.Ltmp120:
	callq	"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
.Ltmp121:
	movb	%al, -18(%rbp)                  # 1-byte Spill
	jmp	.LBB125_3
.LBB125_3:
	movb	-18(%rbp), %al                  # 1-byte Reload
	movb	%al, -17(%rbp)                  # 1-byte Spill
	jmp	.LBB125_4
.LBB125_4:
	movb	-17(%rbp), %al                  # 1-byte Reload
	movb	%al, 127(%rbp)
	movl	$22, 120(%rbp)
	movl	$24, 116(%rbp)
	movq	264(%rbp), %rcx
.Ltmp122:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp123:
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	jmp	.LBB125_5
.LBB125_5:
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rbp)
	movq	72(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	leaq	"?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB"(%rip), %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	callq	"??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	callq	"??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-40(%rbp), %r8                  # 8-byte Reload
	movq	%rax, %rdx
.Ltmp124:
	leaq	80(%rbp), %r9
	callq	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
.Ltmp125:
	jmp	.LBB125_6
.LBB125_6:
	movq	176(%rbp), %rax
	movq	%rax, 64(%rbp)
	movq	192(%rbp), %rdx
	movq	184(%rbp), %rcx
.Ltmp126:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp127:
	movb	%al, -57(%rbp)                  # 1-byte Spill
	jmp	.LBB125_7
.LBB125_7:
	movb	-57(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB125_8
	jmp	.LBB125_18
.LBB125_8:
	movq	184(%rbp), %rcx
.Ltmp128:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp129:
	movb	%al, -58(%rbp)                  # 1-byte Spill
	jmp	.LBB125_9
.LBB125_9:
	movb	-58(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	103(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB125_12
# %bb.10:
	movq	64(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 64(%rbp)
	movb	$43, (%rax)
	movq	184(%rbp), %rcx
.Ltmp134:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp135:
	jmp	.LBB125_11
.LBB125_11:
	jmp	.LBB125_17
.LBB125_12:
	movq	184(%rbp), %rcx
.Ltmp130:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp131:
	movb	%al, -59(%rbp)                  # 1-byte Spill
	jmp	.LBB125_13
.LBB125_13:
	movb	-59(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	102(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB125_16
# %bb.14:
	movq	64(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 64(%rbp)
	movb	$45, (%rax)
	movq	184(%rbp), %rcx
.Ltmp132:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp133:
	jmp	.LBB125_15
.LBB125_15:
	jmp	.LBB125_16
.LBB125_16:
	jmp	.LBB125_17
.LBB125_17:
	jmp	.LBB125_18
.LBB125_18:
	movl	256(%rbp), %eax
	andl	$3584, %eax                     # imm = 0xE00
	movl	%eax, 256(%rbp)
	cmpl	$1024, 256(%rbp)                # imm = 0x400
	jne	.LBB125_20
# %bb.19:
	movl	$8, 60(%rbp)
	jmp	.LBB125_27
.LBB125_20:
	cmpl	$2048, 256(%rbp)                # imm = 0x800
	jne	.LBB125_22
# %bb.21:
	movl	$16, 60(%rbp)
	jmp	.LBB125_26
.LBB125_22:
	cmpl	$0, 256(%rbp)
	jne	.LBB125_24
# %bb.23:
	movl	$0, 60(%rbp)
	jmp	.LBB125_25
.LBB125_24:
	movl	$10, 60(%rbp)
.LBB125_25:
	jmp	.LBB125_26
.LBB125_26:
	jmp	.LBB125_27
.LBB125_27:
	movb	$0, 59(%rbp)
	movb	$0, 58(%rbp)
	movq	192(%rbp), %rdx
	movq	184(%rbp), %rcx
.Ltmp136:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp137:
	movb	%al, -60(%rbp)                  # 1-byte Spill
	jmp	.LBB125_28
.LBB125_28:
	movb	-60(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB125_29
	jmp	.LBB125_46
.LBB125_29:
	movq	184(%rbp), %rcx
.Ltmp138:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp139:
	movb	%al, -61(%rbp)                  # 1-byte Spill
	jmp	.LBB125_30
.LBB125_30:
	movb	-61(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	80(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB125_46
# %bb.31:
	movb	$1, 59(%rbp)
	movq	184(%rbp), %rcx
.Ltmp140:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp141:
	jmp	.LBB125_32
.LBB125_32:
	movq	192(%rbp), %rdx
	movq	184(%rbp), %rcx
.Ltmp142:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp143:
	movb	%al, -62(%rbp)                  # 1-byte Spill
	jmp	.LBB125_33
.LBB125_33:
	movb	-62(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB125_34
	jmp	.LBB125_42
.LBB125_34:
	movq	184(%rbp), %rcx
.Ltmp144:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp145:
	movb	%al, -63(%rbp)                  # 1-byte Spill
	jmp	.LBB125_35
.LBB125_35:
	movb	-63(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	105(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB125_38
# %bb.36:
	movq	184(%rbp), %rcx
.Ltmp146:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp147:
	movb	%al, -64(%rbp)                  # 1-byte Spill
	jmp	.LBB125_37
.LBB125_37:
	movb	-64(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	104(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB125_42
.LBB125_38:
	cmpl	$0, 60(%rbp)
	je	.LBB125_40
# %bb.39:
	cmpl	$16, 60(%rbp)
	jne	.LBB125_42
.LBB125_40:
	movl	$16, 60(%rbp)
	movb	$0, 59(%rbp)
	movq	184(%rbp), %rcx
.Ltmp148:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp149:
	jmp	.LBB125_41
.LBB125_41:
	jmp	.LBB125_45
.LBB125_42:
	cmpl	$0, 60(%rbp)
	jne	.LBB125_44
# %bb.43:
	movl	$8, 60(%rbp)
.LBB125_44:
	jmp	.LBB125_45
.LBB125_45:
	jmp	.LBB125_46
.LBB125_46:
	cmpl	$0, 60(%rbp)
	je	.LBB125_48
# %bb.47:
	cmpl	$10, 60(%rbp)
	jne	.LBB125_49
.LBB125_48:
	movl	$10, %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	jmp	.LBB125_50
.LBB125_49:
	movl	$22, %eax
	movl	$8, %ecx
	cmpl	$8, 60(%rbp)
	cmovel	%ecx, %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
.LBB125_50:
	movl	-68(%rbp), %eax                 # 4-byte Reload
	cltq
	movq	%rax, 48(%rbp)
	movb	59(%rbp), %r8b
	andb	$1, %r8b
.Ltmp150:
	leaq	16(%rbp), %rcx
	movl	$1, %edx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
.Ltmp151:
	jmp	.LBB125_51
.LBB125_51:
	movq	$0, 8(%rbp)
	movq	176(%rbp), %rax
	addq	$31, %rax
	movq	%rax, (%rbp)
.LBB125_52:                             # =>This Inner Loop Header: Depth=1
	movq	192(%rbp), %rdx
	movq	184(%rbp), %rcx
.Ltmp152:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp153:
	movb	%al, -69(%rbp)                  # 1-byte Spill
	jmp	.LBB125_53
.LBB125_53:                             #   in Loop: Header=BB125_52 Depth=1
	movb	-69(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB125_54
	jmp	.LBB125_73
.LBB125_54:                             #   in Loop: Header=BB125_52 Depth=1
	movq	184(%rbp), %rcx
.Ltmp154:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp155:
	movb	%al, -70(%rbp)                  # 1-byte Spill
	jmp	.LBB125_55
.LBB125_55:                             #   in Loop: Header=BB125_52 Depth=1
.Ltmp156:
	movb	-70(%rbp), %dl                  # 1-byte Reload
	leaq	80(%rbp), %rcx
	callq	"??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z"
.Ltmp157:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB125_56
.LBB125_56:                             #   in Loop: Header=BB125_52 Depth=1
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -8(%rbp)
	movq	-8(%rbp), %rax
	cmpq	48(%rbp), %rax
	jae	.LBB125_64
# %bb.57:                               #   in Loop: Header=BB125_52 Depth=1
	movq	-8(%rbp), %rcx
	leaq	"?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	64(%rbp), %rax
	movb	%cl, (%rax)
	testb	$1, 58(%rbp)
	jne	.LBB125_59
# %bb.58:                               #   in Loop: Header=BB125_52 Depth=1
	movq	64(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$48, %eax
	je	.LBB125_61
.LBB125_59:                             #   in Loop: Header=BB125_52 Depth=1
	movq	64(%rbp), %rax
	cmpq	(%rbp), %rax
	jae	.LBB125_61
# %bb.60:                               #   in Loop: Header=BB125_52 Depth=1
	movq	64(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 64(%rbp)
	movb	$1, 58(%rbp)
.LBB125_61:                             #   in Loop: Header=BB125_52 Depth=1
	movb	$1, 59(%rbp)
	movq	8(%rbp), %rdx
	leaq	16(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	je	.LBB125_63
# %bb.62:                               #   in Loop: Header=BB125_52 Depth=1
	movq	8(%rbp), %rdx
	leaq	16(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movb	(%rax), %cl
	addb	$1, %cl
	movb	%cl, (%rax)
.LBB125_63:                             #   in Loop: Header=BB125_52 Depth=1
	jmp	.LBB125_71
.LBB125_64:                             #   in Loop: Header=BB125_52 Depth=1
	movq	8(%rbp), %rdx
	leaq	16(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$0, %eax
	je	.LBB125_68
# %bb.65:                               #   in Loop: Header=BB125_52 Depth=1
	movsbl	127(%rbp), %eax
	cmpl	$0, %eax
	je	.LBB125_68
# %bb.66:                               #   in Loop: Header=BB125_52 Depth=1
	movq	184(%rbp), %rcx
.Ltmp158:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp159:
	movb	%al, -81(%rbp)                  # 1-byte Spill
	jmp	.LBB125_67
.LBB125_67:                             #   in Loop: Header=BB125_52 Depth=1
	movb	-81(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	127(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB125_69
.LBB125_68:
	jmp	.LBB125_73
.LBB125_69:                             #   in Loop: Header=BB125_52 Depth=1
.Ltmp160:
	xorl	%eax, %eax
	movb	%al, %dl
	leaq	16(%rbp), %rcx
	callq	"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
.Ltmp161:
	jmp	.LBB125_70
.LBB125_70:                             #   in Loop: Header=BB125_52 Depth=1
	movq	8(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 8(%rbp)
.LBB125_71:                             #   in Loop: Header=BB125_52 Depth=1
	movq	184(%rbp), %rcx
.Ltmp162:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp163:
	jmp	.LBB125_72
.LBB125_72:                             #   in Loop: Header=BB125_52 Depth=1
	jmp	.LBB125_52
.LBB125_73:
	cmpq	$0, 8(%rbp)
	je	.LBB125_78
# %bb.74:
	movq	8(%rbp), %rdx
	leaq	16(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB125_76
# %bb.75:
	movq	8(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 8(%rbp)
	jmp	.LBB125_77
.LBB125_76:
	movb	$0, 59(%rbp)
.LBB125_77:
	jmp	.LBB125_78
.LBB125_78:
	leaq	128(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	movq	%rax, -16(%rbp)
.LBB125_79:                             # =>This Inner Loop Header: Depth=1
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, 59(%rbp)
	movb	%al, -82(%rbp)                  # 1-byte Spill
	je	.LBB125_81
# %bb.80:                               #   in Loop: Header=BB125_79 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	8(%rbp), %rax
	setb	%al
	movb	%al, -82(%rbp)                  # 1-byte Spill
.LBB125_81:                             #   in Loop: Header=BB125_79 Depth=1
	movb	-82(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB125_82
	jmp	.LBB125_93
.LBB125_82:                             #   in Loop: Header=BB125_79 Depth=1
	movq	-16(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	jne	.LBB125_84
# %bb.83:
	jmp	.LBB125_93
.LBB125_84:                             #   in Loop: Header=BB125_79 Depth=1
	movq	8(%rbp), %rcx
	addq	$-1, %rcx
	movq	%rcx, 8(%rbp)
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	%rcx, %rax
	jae	.LBB125_86
# %bb.85:                               #   in Loop: Header=BB125_79 Depth=1
	movq	-16(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -88(%rbp)                 # 4-byte Spill
	movq	8(%rbp), %rdx
	leaq	16(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	%rax, %rcx
	movl	-88(%rbp), %eax                 # 4-byte Reload
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jne	.LBB125_88
.LBB125_86:                             #   in Loop: Header=BB125_79 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	8(%rbp), %rax
	jne	.LBB125_89
# %bb.87:                               #   in Loop: Header=BB125_79 Depth=1
	movq	-16(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -92(%rbp)                 # 4-byte Spill
	movq	8(%rbp), %rdx
	leaq	16(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	%rax, %rcx
	movl	-92(%rbp), %eax                 # 4-byte Reload
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jge	.LBB125_89
.LBB125_88:                             #   in Loop: Header=BB125_79 Depth=1
	movb	$0, 59(%rbp)
	jmp	.LBB125_92
.LBB125_89:                             #   in Loop: Header=BB125_79 Depth=1
	movq	-16(%rbp), %rax
	movsbl	1(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB125_91
# %bb.90:                               #   in Loop: Header=BB125_79 Depth=1
	movq	-16(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -16(%rbp)
.LBB125_91:                             #   in Loop: Header=BB125_79 Depth=1
	jmp	.LBB125_92
.LBB125_92:                             #   in Loop: Header=BB125_79 Depth=1
	jmp	.LBB125_79
.LBB125_93:
	testb	$1, 59(%rbp)
	je	.LBB125_96
# %bb.94:
	testb	$1, 58(%rbp)
	jne	.LBB125_96
# %bb.95:
	movq	64(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 64(%rbp)
	movb	$48, (%rax)
	jmp	.LBB125_99
.LBB125_96:
	testb	$1, 59(%rbp)
	jne	.LBB125_98
# %bb.97:
	movq	176(%rbp), %rax
	movq	%rax, 64(%rbp)
.LBB125_98:
	jmp	.LBB125_99
.LBB125_99:
	movq	64(%rbp), %rax
	movb	$0, (%rax)
	movl	60(%rbp), %eax
	movl	%eax, -96(%rbp)                 # 4-byte Spill
	leaq	16(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	leaq	128(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movl	-96(%rbp), %eax                 # 4-byte Reload
	addq	$336, %rsp                      # imm = 0x150
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.seh_endproc
	.def	"?dtor$100@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$100@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA":
.seh_proc "?dtor$100@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA"
.LBB125_100:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	16(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.seh_endproc
	.def	"?dtor$101@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$101@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA":
.seh_proc "?dtor$101@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA"
.LBB125_101:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	128(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end29:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.p2align	2
"$cppxdata$?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z")@IMGREL # IPToStateXData
	.long	328                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z":
	.long	-1                              # ToState
	.long	"?dtor$101@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$100@?0??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z@4HA"@IMGREL # Action
"$ip2state$?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z":
	.long	.Lfunc_begin29@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp120@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp152@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp163@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
                                        # -- End function
	.def	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	.globl	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z" # -- Begin function ??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z
	.p2align	4, 0x90
"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z": # @"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.seh_proc "??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	48(%rsp), %rdx
	callq	"?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z"
	andb	$1, %al
	movzbl	%al, %eax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.globl	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z" # -- Begin function ??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z
	.p2align	4, 0x90
"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z": # @"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Lfunc_begin30:
.seh_proc "??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$128, %rsp
	.seh_stackalloc 128
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	leaq	-24(%rbp), %rcx
	xorl	%edx, %edx
	callq	"??0_Lockit@std@@QEAA@H@Z"
	movq	"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB"(%rip), %rax
	movq	%rax, -32(%rbp)
	leaq	"?id@?$numpunct@D@std@@2V0locale@2@A"(%rip), %rcx
	callq	"??Bid@locale@std@@QEAA_KXZ"
	movq	%rax, -40(%rbp)
	movq	-16(%rbp), %rcx
	movq	-40(%rbp), %rdx
.Ltmp164:
	callq	"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
.Ltmp165:
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	jmp	.LBB127_1
.LBB127_1:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -48(%rbp)
	cmpq	$0, -48(%rbp)
	jne	.LBB127_12
# %bb.2:
	cmpq	$0, -32(%rbp)
	je	.LBB127_4
# %bb.3:
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	jmp	.LBB127_11
.LBB127_4:
	movq	-16(%rbp), %rdx
.Ltmp166:
	leaq	-32(%rbp), %rcx
	callq	"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Ltmp167:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB127_5
.LBB127_5:
	movq	-80(%rbp), %rax                 # 8-byte Reload
	cmpq	$-1, %rax
	jne	.LBB127_8
# %bb.6:
.Ltmp170:
	callq	"?_Throw_bad_cast@std@@YAXXZ"
.Ltmp171:
	jmp	.LBB127_7
.LBB127_7:
.LBB127_8:
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rdx
	leaq	-64(%rbp), %rcx
	callq	"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
	movq	-56(%rbp), %rcx
.Ltmp168:
	callq	"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"
.Ltmp169:
	jmp	.LBB127_9
.LBB127_9:
	movq	-56(%rbp), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, "?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB"(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	leaq	-64(%rbp), %rcx
	callq	"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	jmp	.LBB127_11
.LBB127_11:
	jmp	.LBB127_12
.LBB127_12:
	movq	-48(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z")@IMGREL
	.section	.text,"xr",discard,"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$10@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$10@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA"
.LBB127_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$13@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$13@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$13@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA"
.LBB127_13:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end30:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.p2align	2
"$cppxdata$??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z")@IMGREL # IPToStateXData
	.long	120                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z":
	.long	-1                              # ToState
	.long	"?dtor$13@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$10@?0???$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
"$ip2state$??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z":
	.long	.Lfunc_begin30@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp164@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp168@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp169@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
                                        # -- End function
	.def	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*40(%rax)
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
	.globl	"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ" # -- Begin function ?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ
	.p2align	4, 0x90
"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ": # @"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
.seh_proc "?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	cmpq	$0, 16(%rax)
	sete	%al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?thousands_sep@?$numpunct@D@std@@QEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
	.globl	"?thousands_sep@?$numpunct@D@std@@QEBADXZ" # -- Begin function ?thousands_sep@?$numpunct@D@std@@QEBADXZ
	.p2align	4, 0x90
"?thousands_sep@?$numpunct@D@std@@QEBADXZ": # @"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
.seh_proc "?thousands_sep@?$numpunct@D@std@@QEBADXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*32(%rax)
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
	.globl	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z" # -- Begin function ?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z
	.p2align	4, 0x90
"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z": # @"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
.seh_proc "?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r9, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	64(%rsp), %r9
	movq	56(%rsp), %r8
	movq	48(%rsp), %rdx
	movq	(%rcx), %rax
	callq	*56(%rax)
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
	.globl	"??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z" # -- Begin function ??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z
	.p2align	4, 0x90
"??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z": # @"??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
.seh_proc "??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	addq	$27, %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
	.globl	"??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z" # -- Begin function ??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z
	.p2align	4, 0x90
"??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z": # @"??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
.seh_proc "??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	.globl	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z" # -- Begin function ??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z
	.p2align	4, 0x90
"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z": # @"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.seh_proc "??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	48(%rsp), %rdx
	movq	40(%rsp), %rcx
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	xorb	$-1, %al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
	.globl	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ" # -- Begin function ??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ
	.p2align	4, 0x90
"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ": # @"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.seh_proc "??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	testb	$1, 8(%rax)
	jne	.LBB135_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
.LBB135_2:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movb	9(%rax), %al
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	.globl	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ" # -- Begin function ??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ
	.p2align	4, 0x90
"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ": # @"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.seh_proc "??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ"
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	.globl	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z" # -- Begin function ??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z
	.p2align	4, 0x90
"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z": # @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
.Lfunc_begin31:
.seh_proc "??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movb	%r8b, -9(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-32(%rbp), %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	movb	-40(%rbp), %dl
	callq	"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-24(%rbp), %r8
	movb	-9(%rbp), %dl
.Ltmp172:
	callq	"??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z"
.Ltmp173:
	jmp	.LBB137_1
.LBB137_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z")@IMGREL
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z@4HA":
.seh_proc "?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z@4HA"
.LBB137_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end31:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	.p2align	2
"$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z@4HA"@IMGREL # Action
"$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z":
	.long	.Lfunc_begin31@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp172@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp173@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
                                        # -- End function
	.def	"??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z"
	.globl	"??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z" # -- Begin function ??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z
	.p2align	4, 0x90
"??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z": # @"??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z"
.seh_proc "??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rdx
	addq	$26, %rdx
	movq	40(%rsp), %rcx
	leaq	55(%rsp), %r8
	callq	"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
	movq	40(%rsp), %rcx
	subq	%rcx, %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	.globl	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z" # -- Begin function ??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z
	.p2align	4, 0x90
"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z": # @"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
.seh_proc "??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	addq	48(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
	.globl	"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z" # -- Begin function ?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z
	.p2align	4, 0x90
"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z": # @"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
.seh_proc "?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movb	%dl, 87(%rsp)
	movq	%rcx, 72(%rsp)
	movq	72(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	16(%rcx), %rax
	movq	%rax, 64(%rsp)
	movq	64(%rsp), %rax
	cmpq	24(%rcx), %rax
	jae	.LBB140_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	64(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 16(%rcx)
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rcx
	addq	64(%rsp), %rcx
	leaq	87(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	movb	$0, 55(%rsp)
	movq	56(%rsp), %rcx
	movq	64(%rsp), %rax
	addq	$1, %rax
	addq	%rax, %rcx
	leaq	55(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	jmp	.LBB140_3
.LBB140_2:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movb	87(%rsp), %r9b
	movb	48(%rsp), %r8b
	movl	$1, %edx
	callq	"??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z"
.LBB140_3:
	nop
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	.globl	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z" # -- Begin function ??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z
	.p2align	4, 0x90
"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z": # @"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
.seh_proc "??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
	addq	48(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	.globl	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ" # -- Begin function ??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ": # @"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
.seh_proc "??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.globl	"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z" # -- Begin function ?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z
	.p2align	4, 0x90
"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z": # @"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Lfunc_begin32:
.seh_proc "?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$176, %rsp
	.seh_stackalloc 176
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 40(%rbp)
	movq	%rdx, 32(%rbp)
	movq	%rcx, 24(%rbp)
	cmpq	$0, 24(%rbp)
	je	.LBB143_9
# %bb.1:
	movq	24(%rbp), %rax
	cmpq	$0, (%rax)
	jne	.LBB143_9
# %bb.2:
	movl	$48, %ecx
	callq	"??2@YAPEAX_K@Z"
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movb	$1, -81(%rbp)
	movq	32(%rbp), %rcx
	callq	"?_C_str@locale@std@@QEBAPEBDXZ"
	movq	%rax, %rdx
.Ltmp174:
	leaq	-80(%rbp), %rcx
	callq	"??0_Locinfo@std@@QEAA@PEBD@Z"
.Ltmp175:
	jmp	.LBB143_3
.LBB143_3:
.Ltmp176:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	xorl	%eax, %eax
	movl	%eax, %r8d
	leaq	-80(%rbp), %rdx
	movb	$1, %r9b
	callq	"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
.Ltmp177:
	jmp	.LBB143_4
.LBB143_4:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	movb	$0, -81(%rbp)
	movq	24(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	jmp	.LBB143_9
.LBB143_9:
	movl	$4, %eax
	addq	$176, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$5@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$5@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$5@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB143_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$6@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$6@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB143_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	testb	$1, -81(%rbp)
	jne	.LBB143_7
	jmp	.LBB143_8
.LBB143_7:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB143_8:
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end32:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.p2align	2
"$cppxdata$?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # IPToStateXData
	.long	168                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	-1                              # ToState
	.long	"?dtor$6@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$5@?0??_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
"$ip2state$?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	.Lfunc_begin32@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp174@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp176@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp177@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
                                        # -- End function
	.def	"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
	.globl	"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z" # -- Begin function ??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z
	.p2align	4, 0x90
"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z": # @"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
.Lfunc_begin33:
.seh_proc "??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	andb	$1, %r9b
	movb	%r9b, -9(%rbp)
	movq	%r8, -24(%rbp)
	movq	%rdx, -32(%rbp)
	movq	%rcx, -40(%rbp)
	movq	-40(%rbp), %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	movq	-24(%rbp), %rdx
	callq	"??0facet@locale@std@@IEAA@_K@Z"
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	leaq	"??_7?$numpunct@D@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	movb	-9(%rbp), %r8b
	movq	-32(%rbp), %rdx
.Ltmp178:
	andb	$1, %r8b
	callq	"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
.Ltmp179:
	jmp	.LBB144_1
.LBB144_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z")@IMGREL
	.section	.text,"xr",discard,"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z@4HA":
.seh_proc "?dtor$2@?0???0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z@4HA"
.LBB144_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??1facet@locale@std@@MEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end33:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
	.p2align	2
"$cppxdata$??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z@4HA"@IMGREL # Action
"$ip2state$??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z":
	.long	.Lfunc_begin33@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp178@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp179@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0?$numpunct@D@std@@QEAA@AEBV_Locinfo@1@_K_N@Z"
                                        # -- End function
	.def	"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
	.globl	"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z" # -- Begin function ?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z
	.p2align	4, 0x90
"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z": # @"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
.Lfunc_begin34:
.seh_proc "?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$272, %rsp                      # imm = 0x110
	.seh_stackalloc 272
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 136(%rbp)
	andb	$1, %r8b
	movb	%r8b, 135(%rbp)
	movq	%rdx, 120(%rbp)
	movq	%rcx, 112(%rbp)
	movq	112(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	120(%rbp), %rcx
	callq	"?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ"
	movq	%rax, 104(%rbp)
	movq	120(%rbp), %rcx
	leaq	56(%rbp), %rdx
	callq	"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ"
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	$0, 16(%rax)
	movq	$0, 32(%rax)
	movq	$0, 40(%rax)
	movq	%rax, 48(%rbp)
	movq	120(%rbp), %rcx
.Ltmp180:
	movq	%rbp, %rdx
	callq	"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ"
.Ltmp181:
	jmp	.LBB145_1
.LBB145_1:
	testb	$1, 135(%rbp)
	je	.LBB145_3
# %bb.2:
	leaq	"??_C@_00CNPNBAHC@?$AA@"(%rip), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	jmp	.LBB145_4
.LBB145_3:
	movq	104(%rbp), %rax
	movq	16(%rax), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
.LBB145_4:
.Ltmp182:
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	xorl	%eax, %eax
	movl	%eax, %edx
	movq	%rbp, %r8
	callq	"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
.Ltmp183:
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	jmp	.LBB145_5
.LBB145_5:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movq	%rcx, 16(%rax)
	movq	120(%rbp), %rcx
	callq	"?_Getfalse@_Locinfo@std@@QEBAPEBDXZ"
	movq	%rax, %rcx
.Ltmp184:
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	56(%rbp), %r8
	callq	"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
.Ltmp185:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB145_6
.LBB145_6:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	%rcx, 32(%rax)
	movq	120(%rbp), %rcx
	callq	"?_Gettrue@_Locinfo@std@@QEBAPEBDXZ"
	movq	%rax, %rcx
.Ltmp186:
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	56(%rbp), %r8
	callq	"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
.Ltmp187:
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	jmp	.LBB145_7
.LBB145_7:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	-88(%rbp), %rcx                 # 8-byte Reload
	movq	%rcx, 40(%rax)
	movq	$0, 48(%rbp)
	testb	$1, 135(%rbp)
	je	.LBB145_9
# %bb.8:
	movl	$46, %ecx
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	56(%rbp), %r8
	callq	"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
	movb	%al, %cl
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movb	%cl, 24(%rax)
	movl	$44, %ecx
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	56(%rbp), %r8
	callq	"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
	movb	%al, %cl
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movb	%cl, 25(%rax)
	jmp	.LBB145_10
.LBB145_9:
	leaq	-48(%rbp), %rcx
	leaq	56(%rbp), %rdx
	movl	$44, %r8d
	callq	memcpy
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	104(%rbp), %r8
	xorl	%edx, %edx
	leaq	-48(%rbp), %r9
	callq	"??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z"
.LBB145_10:
	leaq	48(%rbp), %rcx
	callq	"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"
	nop
	addq	$272, %rsp                      # imm = 0x110
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z")@IMGREL
	.section	.text,"xr",discard,"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
	.seh_endproc
	.def	"?dtor$11@?0??_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$11@?0??_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z@4HA":
.seh_proc "?dtor$11@?0??_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z@4HA"
.LBB145_11:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	48(%rbp), %rcx
	callq	"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end34:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
	.p2align	2
"$cppxdata$?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z")@IMGREL # IPToStateXData
	.long	264                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z":
	.long	-1                              # ToState
	.long	"?dtor$11@?0??_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z@4HA"@IMGREL # Action
"$ip2state$?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z":
	.long	.Lfunc_begin34@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp180@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp187@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
                                        # -- End function
	.def	"??_G?$numpunct@D@std@@MEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G?$numpunct@D@std@@MEAAPEAXI@Z"
	.globl	"??_G?$numpunct@D@std@@MEAAPEAXI@Z" # -- Begin function ??_G?$numpunct@D@std@@MEAAPEAXI@Z
	.p2align	4, 0x90
"??_G?$numpunct@D@std@@MEAAPEAXI@Z":    # @"??_G?$numpunct@D@std@@MEAAPEAXI@Z"
.seh_proc "??_G?$numpunct@D@std@@MEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1?$numpunct@D@std@@MEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB146_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB146_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_decimal_point@?$numpunct@D@std@@MEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_decimal_point@?$numpunct@D@std@@MEBADXZ"
	.globl	"?do_decimal_point@?$numpunct@D@std@@MEBADXZ" # -- Begin function ?do_decimal_point@?$numpunct@D@std@@MEBADXZ
	.p2align	4, 0x90
"?do_decimal_point@?$numpunct@D@std@@MEBADXZ": # @"?do_decimal_point@?$numpunct@D@std@@MEBADXZ"
.seh_proc "?do_decimal_point@?$numpunct@D@std@@MEBADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	24(%rax), %al
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ"
	.globl	"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ" # -- Begin function ?do_thousands_sep@?$numpunct@D@std@@MEBADXZ
	.p2align	4, 0x90
"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ": # @"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ"
.seh_proc "?do_thousands_sep@?$numpunct@D@std@@MEBADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	25(%rax), %al
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rax
	movq	16(%rax), %rdx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
                                        # kill: def $rcx killed $rax
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rax
	movq	32(%rax), %rdx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
                                        # kill: def $rcx killed $rax
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rax
	movq	40(%rax), %rdx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
                                        # kill: def $rcx killed $rax
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ"
	.globl	"?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ" # -- Begin function ?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ
	.p2align	4, 0x90
"?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ": # @"?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ"
.seh_proc "?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	callq	localeconv
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ"
	.globl	"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ" # -- Begin function ?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ
	.p2align	4, 0x90
"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ": # @"?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ"
.seh_proc "?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movq	%rax, 56(%rsp)
	callq	_Getcvt
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
	.globl	"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z" # -- Begin function ??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z
	.p2align	4, 0x90
"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z": # @"??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
.seh_proc "??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	callq	strlen
	addq	$1, %rax
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rcx
	movl	$1, %edx
	callq	calloc
	movq	%rax, 48(%rsp)
	cmpq	$0, 48(%rsp)
	jne	.LBB154_2
# %bb.1:
	callq	"?_Xbad_alloc@std@@YAXXZ"
.LBB154_2:
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)
.LBB154_3:                              # =>This Inner Loop Header: Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	56(%rsp), %rax
	jae	.LBB154_6
# %bb.4:                                #   in Loop: Header=BB154_3 Depth=1
	movq	64(%rsp), %rax
	movb	(%rax), %cl
	movq	40(%rsp), %rax
	movb	%cl, (%rax)
# %bb.5:                                #   in Loop: Header=BB154_3 Depth=1
	movq	56(%rsp), %rax
	addq	$-1, %rax
	movq	%rax, 56(%rsp)
	movq	40(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 40(%rsp)
	movq	64(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 64(%rsp)
	jmp	.LBB154_3
.LBB154_6:
	movq	48(%rsp), %rax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getfalse@_Locinfo@std@@QEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getfalse@_Locinfo@std@@QEBAPEBDXZ"
	.globl	"?_Getfalse@_Locinfo@std@@QEBAPEBDXZ" # -- Begin function ?_Getfalse@_Locinfo@std@@QEBAPEBDXZ
	.p2align	4, 0x90
"?_Getfalse@_Locinfo@std@@QEBAPEBDXZ":  # @"?_Getfalse@_Locinfo@std@@QEBAPEBDXZ"
.seh_proc "?_Getfalse@_Locinfo@std@@QEBAPEBDXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	leaq	"??_C@_05LAPONLG@false?$AA@"(%rip), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Gettrue@_Locinfo@std@@QEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Gettrue@_Locinfo@std@@QEBAPEBDXZ"
	.globl	"?_Gettrue@_Locinfo@std@@QEBAPEBDXZ" # -- Begin function ?_Gettrue@_Locinfo@std@@QEBAPEBDXZ
	.p2align	4, 0x90
"?_Gettrue@_Locinfo@std@@QEBAPEBDXZ":   # @"?_Gettrue@_Locinfo@std@@QEBAPEBDXZ"
.seh_proc "?_Gettrue@_Locinfo@std@@QEBAPEBDXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	leaq	"??_C@_04LOAJBDKD@true?$AA@"(%rip), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
	.globl	"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z" # -- Begin function ??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z
	.p2align	4, 0x90
"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z": # @"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
.seh_proc "??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%r8, 16(%rsp)
	movq	%rdx, 8(%rsp)
	movb	%cl, 7(%rsp)
	movb	7(%rsp), %al
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z"
	.globl	"??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z" # -- Begin function ??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z
	.p2align	4, 0x90
"??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z": # @"??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z"
.seh_proc "??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r9, 32(%rsp)                   # 8-byte Spill
	movq	%r8, %rax
	movq	32(%rsp), %r8                   # 8-byte Reload
	movq	%rax, 64(%rsp)
	movb	%dl, 63(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rax
	movq	(%rax), %rax
	movb	(%rax), %cl
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
	movq	32(%rsp), %r8                   # 8-byte Reload
	movb	%al, %cl
	movq	40(%rsp), %rax                  # 8-byte Reload
	movb	%cl, 24(%rax)
	movq	64(%rsp), %rax
	movq	8(%rax), %rax
	movb	(%rax), %cl
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
	movb	%al, %cl
	movq	40(%rsp), %rax                  # 8-byte Reload
	movb	%cl, 25(%rax)
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"
	.globl	"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ" # -- Begin function ??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ": # @"??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"
.seh_proc "??1?$_Tidy_guard@V?$numpunct@D@std@@@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	cmpq	$0, (%rax)
	je	.LBB159_2
# %bb.1:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rcx
	callq	"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
.LBB159_2:
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Tidy@?$numpunct@D@std@@AEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	.globl	"?_Tidy@?$numpunct@D@std@@AEAAXXZ" # -- Begin function ?_Tidy@?$numpunct@D@std@@AEAAXXZ
	.p2align	4, 0x90
"?_Tidy@?$numpunct@D@std@@AEAAXXZ":     # @"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
.Lfunc_begin35:
.seh_proc "?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	movq	16(%rax), %rcx
.Ltmp188:
	callq	free
.Ltmp189:
	jmp	.LBB160_1
.LBB160_1:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	32(%rax), %rcx
.Ltmp190:
	callq	free
.Ltmp191:
	jmp	.LBB160_2
.LBB160_2:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	40(%rax), %rcx
.Ltmp192:
	callq	free
.Ltmp193:
	jmp	.LBB160_3
.LBB160_3:
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Tidy@?$numpunct@D@std@@AEAAXXZ")@IMGREL
	.section	.text,"xr",discard,"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	.seh_endproc
	.def	"?dtor$4@?0??_Tidy@?$numpunct@D@std@@AEAAXXZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0??_Tidy@?$numpunct@D@std@@AEAAXXZ@4HA":
.seh_proc "?dtor$4@?0??_Tidy@?$numpunct@D@std@@AEAAXXZ@4HA"
.LBB160_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end35:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	.p2align	2
"$cppxdata$?_Tidy@?$numpunct@D@std@@AEAAXXZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?_Tidy@?$numpunct@D@std@@AEAAXXZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?_Tidy@?$numpunct@D@std@@AEAAXXZ")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Tidy@?$numpunct@D@std@@AEAAXXZ":
	.long	-1                              # ToState
	.long	"?dtor$4@?0??_Tidy@?$numpunct@D@std@@AEAAXXZ@4HA"@IMGREL # Action
"$ip2state$?_Tidy@?$numpunct@D@std@@AEAAXXZ":
	.long	.Lfunc_begin35@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp188@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp193@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
                                        # -- End function
	.def	"??1?$numpunct@D@std@@MEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$numpunct@D@std@@MEAA@XZ"
	.globl	"??1?$numpunct@D@std@@MEAA@XZ"  # -- Begin function ??1?$numpunct@D@std@@MEAA@XZ
	.p2align	4, 0x90
"??1?$numpunct@D@std@@MEAA@XZ":         # @"??1?$numpunct@D@std@@MEAA@XZ"
.seh_proc "??1?$numpunct@D@std@@MEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	leaq	"??_7?$numpunct@D@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	callq	"?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	callq	"??1facet@locale@std@@MEAA@XZ"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
	.globl	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z" # -- Begin function ??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z
	.p2align	4, 0x90
"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z": # @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
.Lfunc_begin36:
.seh_proc "??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rcx
	movq	%rcx, -40(%rbp)                 # 8-byte Spill
	movb	-32(%rbp), %dl
	callq	"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
	movq	-16(%rbp), %rcx
	callq	"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	movq	%rax, %rcx
	callq	"??$_Convert_size@_K_K@std@@YA_K_K@Z"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %r8
	movq	-16(%rbp), %rdx
.Ltmp194:
	callq	"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
.Ltmp195:
	jmp	.LBB162_1
.LBB162_1:
	movq	-40(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z")@IMGREL
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z@4HA":
.seh_proc "?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z@4HA"
.LBB162_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	movq	-40(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end36:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
	.p2align	2
"$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z@4HA"@IMGREL # Action
"$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z":
	.long	.Lfunc_begin36@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp194@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp195@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
                                        # -- End function
	.def	"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
	.globl	"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z" # -- Begin function ??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z
	.p2align	4, 0x90
"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z": # @"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
.seh_proc "??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	callq	"??0?$allocator@D@std@@QEAA@XZ"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
	.globl	"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z" # -- Begin function ??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z
	.p2align	4, 0x90
"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z": # @"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
.seh_proc "??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
# %bb.0:
	subq	$152, %rsp
	.seh_stackalloc 152
	.seh_endprologue
	movq	%r8, 144(%rsp)
	movq	%rdx, 136(%rsp)
	movq	%rcx, 128(%rsp)
	movq	128(%rsp), %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 120(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, %rcx
	movq	56(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	jbe	.LBB164_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB164_2:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 112(%rsp)
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rax
	movq	%rax, 104(%rsp)
	movq	120(%rsp), %r8
	leaq	96(%rsp), %rcx
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rdx
	callq	"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"
	cmpq	$16, 144(%rsp)
	jae	.LBB164_4
# %bb.3:
	movq	144(%rsp), %rcx
	movq	120(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	120(%rsp), %rax
	movq	$15, 24(%rax)
	movq	144(%rsp), %r8
	movq	136(%rsp), %rdx
	movq	120(%rsp), %rcx
	callq	"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	$0, 95(%rsp)
	movq	120(%rsp), %rcx
	addq	144(%rsp), %rcx
	leaq	95(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	leaq	96(%rsp), %rcx
	callq	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
	jmp	.LBB164_5
.LBB164_4:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	120(%rsp), %rax
	movq	$15, 24(%rax)
	movq	144(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rcx
	movq	80(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 72(%rsp)
	movq	120(%rsp), %rcx
	leaq	72(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
	movq	144(%rsp), %rcx
	movq	120(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	80(%rsp), %rcx
	movq	120(%rsp), %rax
	movq	%rcx, 24(%rax)
	movq	144(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	136(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	32(%rsp), %rdx                  # 8-byte Reload
	movq	40(%rsp), %r8                   # 8-byte Reload
	movq	%rax, %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	$0, 71(%rsp)
	movq	72(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, %rcx
	addq	144(%rsp), %rcx
	leaq	71(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	leaq	96(%rsp), %rcx
	callq	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
.LBB164_5:
	nop
	addq	$152, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Convert_size@_K_K@std@@YA_K_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Convert_size@_K_K@std@@YA_K_K@Z"
	.globl	"??$_Convert_size@_K_K@std@@YA_K_K@Z" # -- Begin function ??$_Convert_size@_K_K@std@@YA_K_K@Z
	.p2align	4, 0x90
"??$_Convert_size@_K_K@std@@YA_K_K@Z":  # @"??$_Convert_size@_K_K@std@@YA_K_K@Z"
.seh_proc "??$_Convert_size@_K_K@std@@YA_K_K@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	.globl	"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z" # -- Begin function ?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z
	.p2align	4, 0x90
"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z": # @"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
.Lfunc_begin37:
.seh_proc "?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rcx
.Ltmp196:
	callq	strlen
.Ltmp197:
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	jmp	.LBB166_1
.LBB166_1:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z")@IMGREL
	.section	.text,"xr",discard,"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	.seh_endproc
	.def	"?dtor$2@?0??length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0??length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z@4HA":
.seh_proc "?dtor$2@?0??length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z@4HA"
.LBB166_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end37:
	.seh_handlerdata
	.section	.text,"xr",discard,"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	.p2align	2
"$cppxdata$?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0??length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z@4HA"@IMGREL # Action
"$ip2state$?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z":
	.long	.Lfunc_begin37@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp196@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp197@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
                                        # -- End function
	.def	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
	.globl	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ" # -- Begin function ??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ": # @"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
.seh_proc "??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$allocator@D@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$allocator@D@std@@QEAA@XZ"
	.globl	"??0?$allocator@D@std@@QEAA@XZ" # -- Begin function ??0?$allocator@D@std@@QEAA@XZ
	.p2align	4, 0x90
"??0?$allocator@D@std@@QEAA@XZ":        # @"??0?$allocator@D@std@@QEAA@XZ"
.seh_proc "??0?$allocator@D@std@@QEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
	.globl	"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" # -- Begin function ??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ": # @"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
.seh_proc "??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	$0, 16(%rax)
	movq	$0, 24(%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
	.globl	"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" # -- Begin function ??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ": # @"??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
.seh_proc "??0_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	xorl	%edx, %edx
	movl	$16, %r8d
	callq	memset
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	.globl	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ" # -- Begin function ?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ
	.p2align	4, 0x90
"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ": # @"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
.seh_proc "?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rcx, 80(%rsp)
	movq	80(%rsp), %rcx
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"
	movq	%rax, %rcx
	callq	"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"
	movq	%rax, 72(%rsp)
	movq	$16, 56(%rsp)
	leaq	72(%rsp), %rcx
	leaq	56(%rsp), %rdx
	callq	"??$max@_K@std@@YAAEB_KAEB_K0@Z"
	movq	(%rax), %rax
	movq	%rax, 64(%rsp)
	movq	64(%rsp), %rax
	subq	$1, %rax
	movq	%rax, 48(%rsp)
	callq	"?max@?$numeric_limits@_J@std@@SA_JXZ"
	movq	%rax, 40(%rsp)
	leaq	40(%rsp), %rcx
	leaq	48(%rsp), %rdx
	callq	"??$min@_K@std@@YAAEB_KAEB_K0@Z"
	movq	(%rax), %rax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Xlen_string@std@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Xlen_string@std@@YAXXZ"
	.globl	"?_Xlen_string@std@@YAXXZ"      # -- Begin function ?_Xlen_string@std@@YAXXZ
	.p2align	4, 0x90
"?_Xlen_string@std@@YAXXZ":             # @"?_Xlen_string@std@@YAXXZ"
.seh_proc "?_Xlen_string@std@@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	leaq	"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@"(%rip), %rcx
	callq	"?_Xlength_error@std@@YAXPEBD@Z"
	int3
	.seh_endproc
                                        # -- End function
	.def	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	.globl	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ" # -- Begin function ?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ
	.p2align	4, 0x90
"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ": # @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
.seh_proc "?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"
	.globl	"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z" # -- Begin function ??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z
	.p2align	4, 0x90
"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z": # @"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"
.seh_proc "??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%r8, 16(%rsp)
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	.globl	"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z" # -- Begin function ?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z
	.p2align	4, 0x90
"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z": # @"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
.seh_proc "?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%r8, 48(%rsp)
	movq	%rdx, 40(%rsp)
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	movq	40(%rsp), %rdx
	movq	48(%rsp), %r8
	shlq	$0, %r8
	callq	memmove
	movq	32(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	.globl	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z" # -- Begin function ?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z
	.p2align	4, 0x90
"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z": # @"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
.seh_proc "?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	movq	8(%rsp), %rax
	movb	(%rax), %cl
	movq	(%rsp), %rax
	movb	%cl, (%rax)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
	.globl	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ" # -- Begin function ?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ
	.p2align	4, 0x90
"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ": # @"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
.seh_proc "?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	.globl	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z" # -- Begin function ?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z
	.p2align	4, 0x90
"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z": # @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
.seh_proc "?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, %r8
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	24(%rax), %rdx
	movq	48(%rsp), %rcx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	.globl	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z" # -- Begin function ?allocate@?$allocator@D@std@@QEAAPEAD_K@Z
	.p2align	4, 0x90
"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z": # @"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
.seh_proc "?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	48(%rsp), %rcx
	callq	"??$_Get_size_of_n@$00@std@@YA_K_K@Z"
	movq	%rax, %rcx
	callq	"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
	.globl	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z" # -- Begin function ??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z
	.p2align	4, 0x90
"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z": # @"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
.seh_proc "??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	callq	"??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z"
	movq	48(%rsp), %rcx
	movq	(%rcx), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	.globl	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z" # -- Begin function ?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z
	.p2align	4, 0x90
"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z": # @"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
.seh_proc "?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%r8, 48(%rsp)
	movq	%rdx, 40(%rsp)
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	movq	40(%rsp), %rdx
	movq	48(%rsp), %r8
	shlq	$0, %r8
	callq	memcpy
	movq	32(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Unfancy@D@std@@YAPEADPEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	.globl	"??$_Unfancy@D@std@@YAPEADPEAD@Z" # -- Begin function ??$_Unfancy@D@std@@YAPEADPEAD@Z
	.p2align	4, 0x90
"??$_Unfancy@D@std@@YAPEADPEAD@Z":      # @"??$_Unfancy@D@std@@YAPEADPEAD@Z"
.seh_proc "??$_Unfancy@D@std@@YAPEADPEAD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"
	.globl	"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z" # -- Begin function ?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z
	.p2align	4, 0x90
"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z": # @"?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"
.seh_proc "?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	$-1, %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"
	.globl	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ" # -- Begin function ?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ
	.p2align	4, 0x90
"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ": # @"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"
.seh_proc "?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$max@_K@std@@YAAEB_KAEB_K0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$max@_K@std@@YAAEB_KAEB_K0@Z"
	.globl	"??$max@_K@std@@YAAEB_KAEB_K0@Z" # -- Begin function ??$max@_K@std@@YAAEB_KAEB_K0@Z
	.p2align	4, 0x90
"??$max@_K@std@@YAAEB_KAEB_K0@Z":       # @"??$max@_K@std@@YAAEB_KAEB_K0@Z"
.seh_proc "??$max@_K@std@@YAAEB_KAEB_K0@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rdx, 16(%rsp)
	movq	%rcx, 8(%rsp)
	movq	8(%rsp), %rax
	movq	(%rax), %rax
	movq	16(%rsp), %rcx
	cmpq	(%rcx), %rax
	jae	.LBB185_2
# %bb.1:
	movq	16(%rsp), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	jmp	.LBB185_3
.LBB185_2:
	movq	8(%rsp), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
.LBB185_3:
	movq	(%rsp), %rax                    # 8-byte Reload
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$min@_K@std@@YAAEB_KAEB_K0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$min@_K@std@@YAAEB_KAEB_K0@Z"
	.globl	"??$min@_K@std@@YAAEB_KAEB_K0@Z" # -- Begin function ??$min@_K@std@@YAAEB_KAEB_K0@Z
	.p2align	4, 0x90
"??$min@_K@std@@YAAEB_KAEB_K0@Z":       # @"??$min@_K@std@@YAAEB_KAEB_K0@Z"
.seh_proc "??$min@_K@std@@YAAEB_KAEB_K0@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rdx, 16(%rsp)
	movq	%rcx, 8(%rsp)
	movq	16(%rsp), %rax
	movq	(%rax), %rax
	movq	8(%rsp), %rcx
	cmpq	(%rcx), %rax
	jae	.LBB186_2
# %bb.1:
	movq	16(%rsp), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
	jmp	.LBB186_3
.LBB186_2:
	movq	8(%rsp), %rax
	movq	%rax, (%rsp)                    # 8-byte Spill
.LBB186_3:
	movq	(%rsp), %rax                    # 8-byte Reload
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?max@?$numeric_limits@_J@std@@SA_JXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?max@?$numeric_limits@_J@std@@SA_JXZ"
	.globl	"?max@?$numeric_limits@_J@std@@SA_JXZ" # -- Begin function ?max@?$numeric_limits@_J@std@@SA_JXZ
	.p2align	4, 0x90
"?max@?$numeric_limits@_J@std@@SA_JXZ": # @"?max@?$numeric_limits@_J@std@@SA_JXZ"
# %bb.0:
	movabsq	$9223372036854775807, %rax      # imm = 0x7FFFFFFFFFFFFFFF
	retq
                                        # -- End function
	.def	"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"
	.globl	"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ" # -- Begin function ?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ
	.p2align	4, 0x90
"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ": # @"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"
.seh_proc "?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"
	.globl	"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ" # -- Begin function ?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ
	.p2align	4, 0x90
"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ": # @"?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"
.seh_proc "?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"
	.globl	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z" # -- Begin function ?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z
	.p2align	4, 0x90
"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z": # @"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"
.seh_proc "?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rax
	orq	$15, %rax
	movq	%rax, 48(%rsp)
	movq	48(%rsp), %rax
	cmpq	72(%rsp), %rax
	jbe	.LBB190_2
# %bb.1:
	movq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	jmp	.LBB190_5
.LBB190_2:
	movq	64(%rsp), %rax
	movq	72(%rsp), %rcx
	movq	64(%rsp), %rdx
	shrq	$1, %rdx
	subq	%rdx, %rcx
	cmpq	%rcx, %rax
	jbe	.LBB190_4
# %bb.3:
	movq	72(%rsp), %rax
	movq	%rax, 80(%rsp)
	jmp	.LBB190_5
.LBB190_4:
	movq	64(%rsp), %rax
	movq	64(%rsp), %rcx
	shrq	$1, %rcx
	addq	%rcx, %rax
	movq	%rax, 40(%rsp)
	leaq	48(%rsp), %rcx
	leaq	40(%rsp), %rdx
	callq	"??$max@_K@std@@YAAEB_KAEB_K0@Z"
	movq	(%rax), %rax
	movq	%rax, 80(%rsp)
.LBB190_5:
	movq	80(%rsp), %rax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"
	.globl	"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z" # -- Begin function ??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z
	.p2align	4, 0x90
"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z": # @"??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"
.seh_proc "??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 40(%rsp)
	cmpq	$4096, 40(%rsp)                 # imm = 0x1000
	jb	.LBB191_2
# %bb.1:
	movq	40(%rsp), %rcx
	callq	"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"
	movq	%rax, 48(%rsp)
	jmp	.LBB191_5
.LBB191_2:
	cmpq	$0, 40(%rsp)
	je	.LBB191_4
# %bb.3:
	movq	40(%rsp), %rcx
	callq	"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"
	movq	%rax, 48(%rsp)
	jmp	.LBB191_5
.LBB191_4:
	movq	$0, 48(%rsp)
.LBB191_5:
	movq	48(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Get_size_of_n@$00@std@@YA_K_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Get_size_of_n@$00@std@@YA_K_K@Z"
	.globl	"??$_Get_size_of_n@$00@std@@YA_K_K@Z" # -- Begin function ??$_Get_size_of_n@$00@std@@YA_K_K@Z
	.p2align	4, 0x90
"??$_Get_size_of_n@$00@std@@YA_K_K@Z":  # @"??$_Get_size_of_n@$00@std@@YA_K_K@Z"
.seh_proc "??$_Get_size_of_n@$00@std@@YA_K_K@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rcx, 8(%rsp)
	movb	$0, 7(%rsp)
	movq	8(%rsp), %rax
	shlq	$0, %rax
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"
	.globl	"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z" # -- Begin function ??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z
	.p2align	4, 0x90
"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z": # @"??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"
.seh_proc "??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rax
	addq	$39, %rax
	movq	%rax, 56(%rsp)
	movq	56(%rsp), %rax
	cmpq	64(%rsp), %rax
	ja	.LBB193_2
# %bb.1:
	callq	"?_Throw_bad_array_new_length@std@@YAXXZ"
.LBB193_2:
	movq	56(%rsp), %rcx
	callq	"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"
	movq	%rax, 48(%rsp)
# %bb.3:
	cmpq	$0, 48(%rsp)
	je	.LBB193_5
# %bb.4:
	jmp	.LBB193_7
.LBB193_5:
	jmp	.LBB193_6
.LBB193_6:
	callq	_invalid_parameter_noinfo_noreturn
.LBB193_7:
	jmp	.LBB193_8
.LBB193_8:
	movq	48(%rsp), %rax
	addq	$39, %rax
	andq	$-32, %rax
	movq	%rax, 40(%rsp)
	movq	48(%rsp), %rcx
	movq	40(%rsp), %rax
	movq	%rcx, -8(%rax)
	movq	40(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"
	.globl	"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z" # -- Begin function ?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z
	.p2align	4, 0x90
"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z": # @"?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"
.seh_proc "?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??2@YAPEAX_K@Z"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Throw_bad_array_new_length@std@@YAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Throw_bad_array_new_length@std@@YAXXZ"
	.globl	"?_Throw_bad_array_new_length@std@@YAXXZ" # -- Begin function ?_Throw_bad_array_new_length@std@@YAXXZ
	.p2align	4, 0x90
"?_Throw_bad_array_new_length@std@@YAXXZ": # @"?_Throw_bad_array_new_length@std@@YAXXZ"
.seh_proc "?_Throw_bad_array_new_length@std@@YAXXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	leaq	32(%rsp), %rcx
	callq	"??0bad_array_new_length@std@@QEAA@XZ"
	leaq	32(%rsp), %rcx
	leaq	"_TI3?AVbad_array_new_length@std@@"(%rip), %rdx
	callq	_CxxThrowException
	int3
	.seh_endproc
                                        # -- End function
	.def	"??0bad_array_new_length@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0bad_array_new_length@std@@QEAA@XZ"
	.globl	"??0bad_array_new_length@std@@QEAA@XZ" # -- Begin function ??0bad_array_new_length@std@@QEAA@XZ
	.p2align	4, 0x90
"??0bad_array_new_length@std@@QEAA@XZ": # @"??0bad_array_new_length@std@@QEAA@XZ"
.seh_proc "??0bad_array_new_length@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	leaq	"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@"(%rip), %rdx
	callq	"??0bad_alloc@std@@AEAA@QEBD@Z"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7bad_array_new_length@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0bad_array_new_length@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0bad_array_new_length@std@@QEAA@AEBV01@@Z"
	.globl	"??0bad_array_new_length@std@@QEAA@AEBV01@@Z" # -- Begin function ??0bad_array_new_length@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0bad_array_new_length@std@@QEAA@AEBV01@@Z": # @"??0bad_array_new_length@std@@QEAA@AEBV01@@Z"
.seh_proc "??0bad_array_new_length@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0bad_alloc@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7bad_array_new_length@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0bad_alloc@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0bad_alloc@std@@QEAA@AEBV01@@Z"
	.globl	"??0bad_alloc@std@@QEAA@AEBV01@@Z" # -- Begin function ??0bad_alloc@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0bad_alloc@std@@QEAA@AEBV01@@Z":     # @"??0bad_alloc@std@@QEAA@AEBV01@@Z"
.seh_proc "??0bad_alloc@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0exception@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7bad_alloc@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1bad_array_new_length@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1bad_array_new_length@std@@UEAA@XZ"
	.globl	"??1bad_array_new_length@std@@UEAA@XZ" # -- Begin function ??1bad_array_new_length@std@@UEAA@XZ
	.p2align	4, 0x90
"??1bad_array_new_length@std@@UEAA@XZ": # @"??1bad_array_new_length@std@@UEAA@XZ"
.seh_proc "??1bad_array_new_length@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1bad_alloc@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0bad_alloc@std@@AEAA@QEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0bad_alloc@std@@AEAA@QEBD@Z"
	.globl	"??0bad_alloc@std@@AEAA@QEBD@Z" # -- Begin function ??0bad_alloc@std@@AEAA@QEBD@Z
	.p2align	4, 0x90
"??0bad_alloc@std@@AEAA@QEBD@Z":        # @"??0bad_alloc@std@@AEAA@QEBD@Z"
.seh_proc "??0bad_alloc@std@@AEAA@QEBD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	movl	$1, %r8d
	callq	"??0exception@std@@QEAA@QEBDH@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7bad_alloc@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gbad_array_new_length@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gbad_array_new_length@std@@UEAAPEAXI@Z"
	.globl	"??_Gbad_array_new_length@std@@UEAAPEAXI@Z" # -- Begin function ??_Gbad_array_new_length@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gbad_array_new_length@std@@UEAAPEAXI@Z": # @"??_Gbad_array_new_length@std@@UEAAPEAXI@Z"
.seh_proc "??_Gbad_array_new_length@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1bad_array_new_length@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB201_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB201_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gbad_alloc@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gbad_alloc@std@@UEAAPEAXI@Z"
	.globl	"??_Gbad_alloc@std@@UEAAPEAXI@Z" # -- Begin function ??_Gbad_alloc@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gbad_alloc@std@@UEAAPEAXI@Z":       # @"??_Gbad_alloc@std@@UEAAPEAXI@Z"
.seh_proc "??_Gbad_alloc@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1bad_alloc@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB202_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB202_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1bad_alloc@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1bad_alloc@std@@UEAA@XZ"
	.globl	"??1bad_alloc@std@@UEAA@XZ"     # -- Begin function ??1bad_alloc@std@@UEAA@XZ
	.p2align	4, 0x90
"??1bad_alloc@std@@UEAA@XZ":            # @"??1bad_alloc@std@@UEAA@XZ"
.seh_proc "??1bad_alloc@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1exception@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z"
	.globl	"??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z" # -- Begin function ??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z
	.p2align	4, 0x90
"??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z": # @"??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z"
.seh_proc "??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
	.globl	"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" # -- Begin function ??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ": # @"??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
.seh_proc "??1?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
	.globl	"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ" # -- Begin function ??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ
	.p2align	4, 0x90
"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ": # @"??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
.seh_proc "??1_Bxty@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
	.globl	"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ" # -- Begin function ?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ
	.p2align	4, 0x90
"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ": # @"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
.seh_proc "?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movb	$1, %al
	cmpq	$0, (%rcx)
	movb	%al, 55(%rsp)                   # 1-byte Spill
	je	.LBB207_2
# %bb.1:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rcx
	callq	"?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	movl	%eax, 60(%rsp)
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, 56(%rsp)
	leaq	56(%rsp), %rcx
	leaq	60(%rsp), %rdx
	callq	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	movb	%al, 55(%rsp)                   # 1-byte Spill
.LBB207_2:
	movb	55(%rsp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB207_3
	jmp	.LBB207_4
.LBB207_3:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	$0, (%rax)
	jmp	.LBB207_5
.LBB207_4:
	leaq	60(%rsp), %rcx
	callq	"?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z"
	movb	%al, %cl
	movq	40(%rsp), %rax                  # 8-byte Reload
	movb	%cl, 9(%rax)
.LBB207_5:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movb	$1, 8(%rax)
	movb	9(%rax), %al
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ"
	.globl	"?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ" # -- Begin function ?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ
	.p2align	4, 0x90
"?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ": # @"?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ"
.seh_proc "?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movb	$1, %al
	cmpq	$0, (%rcx)
	movb	%al, 55(%rsp)                   # 1-byte Spill
	je	.LBB208_2
# %bb.1:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rcx
	callq	"?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	movl	%eax, 60(%rsp)
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, 56(%rsp)
	leaq	56(%rsp), %rcx
	leaq	60(%rsp), %rdx
	callq	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	movb	%al, 55(%rsp)                   # 1-byte Spill
.LBB208_2:
	movb	55(%rsp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB208_3
	jmp	.LBB208_4
.LBB208_3:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	$0, (%rax)
	movb	$1, 8(%rax)
	jmp	.LBB208_5
.LBB208_4:
	movq	40(%rsp), %rax                  # 8-byte Reload
	movb	$0, 8(%rax)
.LBB208_5:
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z"
	.globl	"??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z" # -- Begin function ??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z
	.p2align	4, 0x90
"??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z": # @"??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z"
.seh_proc "??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z"
# %bb.0:
	subq	$152, %rsp
	.seh_stackalloc 152
	.seh_endprologue
	movq	%r8, 144(%rsp)
	movb	%dl, 143(%rsp)
	movq	%rcx, 128(%rsp)
	movq	128(%rsp), %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 120(%rsp)
	movq	144(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, %rcx
	movq	56(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	jbe	.LBB209_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB209_2:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 112(%rsp)
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rax
	movq	%rax, 104(%rsp)
	movq	120(%rsp), %r8
	leaq	96(%rsp), %rcx
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rdx
	callq	"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"
	cmpq	$16, 144(%rsp)
	jae	.LBB209_4
# %bb.3:
	movq	144(%rsp), %rcx
	movq	120(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	120(%rsp), %rax
	movq	$15, 24(%rax)
	movb	143(%rsp), %r8b
	movq	144(%rsp), %rdx
	movq	120(%rsp), %rcx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	movb	$0, 95(%rsp)
	movq	120(%rsp), %rcx
	addq	144(%rsp), %rcx
	leaq	95(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	leaq	96(%rsp), %rcx
	callq	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
	jmp	.LBB209_5
.LBB209_4:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	120(%rsp), %rax
	movq	$15, 24(%rax)
	movq	144(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	%rax, 80(%rsp)
	movq	112(%rsp), %rcx
	movq	80(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 72(%rsp)
	movq	120(%rsp), %rcx
	leaq	72(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
	movq	144(%rsp), %rcx
	movq	120(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	80(%rsp), %rcx
	movq	120(%rsp), %rax
	movq	%rcx, 24(%rax)
	movb	143(%rsp), %al
	movb	%al, 47(%rsp)                   # 1-byte Spill
	movq	144(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	32(%rsp), %rdx                  # 8-byte Reload
	movb	47(%rsp), %r8b                  # 1-byte Reload
	movq	%rax, %rcx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	movb	$0, 71(%rsp)
	movq	72(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, %rcx
	addq	144(%rsp), %rcx
	leaq	71(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	leaq	96(%rsp), %rcx
	callq	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
.LBB209_5:
	nop
	addq	$152, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	.globl	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z" # -- Begin function ?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z
	.p2align	4, 0x90
"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z": # @"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
.seh_proc "?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	%r8b, 71(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movsbl	71(%rsp), %eax
	movb	%al, %dl
	movq	56(%rsp), %r8
	callq	memset
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
	.globl	"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z" # -- Begin function ??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z
	.p2align	4, 0x90
"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z": # @"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
.seh_proc "??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 72(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	72(%rsp), %rcx
	callq	"??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z"
	testb	$1, %al
	jne	.LBB211_2
# %bb.1:
	movq	64(%rsp), %rax
	movq	%rax, 80(%rsp)
	jmp	.LBB211_3
.LBB211_2:
	leaq	56(%rsp), %rcx
	callq	"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z"
	movq	%rax, 48(%rsp)
	movq	72(%rsp), %rax
	movb	(%rax), %al
	movb	%al, 39(%rsp)                   # 1-byte Spill
	leaq	64(%rsp), %rcx
	callq	"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z"
	movb	39(%rsp), %r8b                  # 1-byte Reload
	movq	%rax, %rdx
	movq	48(%rsp), %rcx
	callq	"??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z"
	movq	%rax, 40(%rsp)
	movq	40(%rsp), %rax
	movq	%rax, 80(%rsp)
.LBB211_3:
	movq	80(%rsp), %rax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z"
	.globl	"??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z" # -- Begin function ??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z
	.p2align	4, 0x90
"??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z": # @"??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z"
.seh_proc "??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rcx, 8(%rsp)
	movb	$-128, 7(%rsp)
	movb	$127, 6(%rsp)
	movq	8(%rsp), %rax
	movsbl	(%rax), %edx
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	movl	$4294967168, %ecx               # imm = 0xFFFFFF80
	cmpl	%edx, %ecx
	movb	%al, 5(%rsp)                    # 1-byte Spill
	jg	.LBB212_2
# %bb.1:
	movq	8(%rsp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	setle	%al
	movb	%al, 5(%rsp)                    # 1-byte Spill
.LBB212_2:
	movb	5(%rsp), %al                    # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z"
	.globl	"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z" # -- Begin function ??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z
	.p2align	4, 0x90
"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z": # @"??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z"
.seh_proc "??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z"
	.globl	"??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z" # -- Begin function ??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z
	.p2align	4, 0x90
"??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z": # @"??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z"
.seh_proc "??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%r8b, 55(%rsp)
	movq	%rdx, 40(%rsp)
	movq	%rcx, 32(%rsp)
	movb	55(%rsp), %r8b
	movq	40(%rsp), %rdx
	movq	32(%rsp), %rcx
	callq	__std_find_trivial_1
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	.globl	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ" # -- Begin function ?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ
	.p2align	4, 0x90
"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ": # @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
.seh_proc "?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 40(%rsp)
	callq	"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB215_1
	jmp	.LBB215_2
.LBB215_1:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, 40(%rsp)
.LBB215_2:
	movq	40(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
	.globl	"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ" # -- Begin function ?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ
	.p2align	4, 0x90
"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ": # @"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
.seh_proc "?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rcx
	movl	$16, %eax
	cmpq	24(%rcx), %rax
	setbe	%al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z"
	.globl	"??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z" # -- Begin function ??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z
	.p2align	4, 0x90
"??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z": # @"??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z"
.seh_proc "??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z"
# %bb.0:
	subq	$168, %rsp
	.seh_stackalloc 168
	.seh_endprologue
	movb	%r8b, 160(%rsp)
	movb	%r9b, 159(%rsp)
	movq	%rdx, 144(%rsp)
	movq	%rcx, 136(%rsp)
	movq	136(%rsp), %rcx
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	%rcx, 128(%rsp)
	movq	128(%rsp), %rax
	movq	16(%rax), %rax
	movq	%rax, 120(%rsp)
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	subq	120(%rsp), %rax
	cmpq	144(%rsp), %rax
	jae	.LBB217_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB217_2:
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movq	120(%rsp), %rax
	addq	144(%rsp), %rax
	movq	%rax, 112(%rsp)
	movq	128(%rsp), %rax
	movq	24(%rax), %rax
	movq	%rax, 104(%rsp)
	movq	112(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 96(%rsp)
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 88(%rsp)
	movq	88(%rsp), %rcx
	movq	96(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 80(%rsp)
	movq	128(%rsp), %rcx
	callq	"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	movq	112(%rsp), %rcx
	movq	128(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	96(%rsp), %rcx
	movq	128(%rsp), %rax
	movq	%rcx, 24(%rax)
	movq	80(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, 72(%rsp)
	movl	$16, %eax
	cmpq	104(%rsp), %rax
	ja	.LBB217_4
# %bb.3:
	movq	128(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 64(%rsp)
	movb	159(%rsp), %al
	movb	%al, 55(%rsp)                   # 1-byte Spill
	movq	120(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	40(%rsp), %r9                   # 8-byte Reload
	movq	%rax, %r8
	movb	55(%rsp), %al                   # 1-byte Reload
	movq	72(%rsp), %rdx
	leaq	160(%rsp), %rcx
	movb	%al, 32(%rsp)
	callq	"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z"
	movq	88(%rsp), %rcx
	movq	104(%rsp), %r8
	addq	$1, %r8
	movq	64(%rsp), %rdx
	callq	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
	movq	80(%rsp), %rcx
	movq	128(%rsp), %rax
	movq	%rcx, (%rax)
	jmp	.LBB217_5
.LBB217_4:
	movb	159(%rsp), %al
	movq	120(%rsp), %r9
	movq	128(%rsp), %r8
	movq	72(%rsp), %rdx
	leaq	160(%rsp), %rcx
	movb	%al, 32(%rsp)
	callq	"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z"
	movq	128(%rsp), %rcx
	leaq	80(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
.LBB217_5:
	movq	56(%rsp), %rax                  # 8-byte Reload
	addq	$168, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Orphan_all@_Container_base0@std@@QEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	.globl	"?_Orphan_all@_Container_base0@std@@QEAAXXZ" # -- Begin function ?_Orphan_all@_Container_base0@std@@QEAAXXZ
	.p2align	4, 0x90
"?_Orphan_all@_Container_base0@std@@QEAAXXZ": # @"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
.seh_proc "?_Orphan_all@_Container_base0@std@@QEAAXXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z"
	.globl	"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z" # -- Begin function ??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z
	.p2align	4, 0x90
"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z": # @"??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z"
.seh_proc "??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	112(%rsp), %al
	movq	%r9, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	64(%rsp), %r8
	movq	56(%rsp), %rdx
	movq	48(%rsp), %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movq	48(%rsp), %rcx
	addq	64(%rsp), %rcx
	leaq	112(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	movb	$0, 39(%rsp)
	movq	48(%rsp), %rcx
	movq	64(%rsp), %rax
	addq	$1, %rax
	addq	%rax, %rcx
	leaq	39(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
	.globl	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z" # -- Begin function ?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z
	.p2align	4, 0x90
"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z": # @"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
.seh_proc "?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%r8, 48(%rsp)
	movq	%rdx, 40(%rsp)
	movq	%rcx, 32(%rsp)
	movq	48(%rsp), %rdx
	shlq	$0, %rdx
	movq	40(%rsp), %rcx
	callq	"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	.globl	"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z" # -- Begin function ??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z
	.p2align	4, 0x90
"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z": # @"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
.Lfunc_begin38:
.seh_proc "??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	64(%rsp), %rbp
	.seh_setframe %rbp, 64
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	cmpq	$4096, -16(%rbp)                # imm = 0x1000
	jb	.LBB221_3
# %bb.1:
.Ltmp198:
	leaq	-24(%rbp), %rcx
	leaq	-16(%rbp), %rdx
	callq	"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"
.Ltmp199:
	jmp	.LBB221_2
.LBB221_2:
	jmp	.LBB221_3
.LBB221_3:
	movq	-16(%rbp), %rdx
	movq	-24(%rbp), %rcx
	callq	"??3@YAXPEAX_K@Z"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z")@IMGREL
	.section	.text,"xr",discard,"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	.seh_endproc
	.def	"?dtor$4@?0???$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$4@?0???$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z@4HA":
.seh_proc "?dtor$4@?0???$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z@4HA"
.LBB221_4:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	64(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end38:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	.p2align	2
"$cppxdata$??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z")@IMGREL # IPToStateXData
	.long	56                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z":
	.long	-1                              # ToState
	.long	"?dtor$4@?0???$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z@4HA"@IMGREL # Action
"$ip2state$??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z":
	.long	.Lfunc_begin38@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp198@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp199@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
                                        # -- End function
	.def	"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"
	.globl	"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z" # -- Begin function ?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z
	.p2align	4, 0x90
"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z": # @"?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"
.seh_proc "?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rdx, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	80(%rsp), %rax
	movq	(%rax), %rcx
	addq	$39, %rcx
	movq	%rcx, (%rax)
	movq	72(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 64(%rsp)
	movq	64(%rsp), %rax
	movq	-8(%rax), %rax
	movq	%rax, 56(%rsp)
	movq	$8, 48(%rsp)
	movq	72(%rsp), %rax
	movq	(%rax), %rax
	subq	56(%rsp), %rax
	movq	%rax, 40(%rsp)
# %bb.1:
	cmpq	$8, 40(%rsp)
	jb	.LBB222_4
# %bb.2:
	cmpq	$39, 40(%rsp)
	ja	.LBB222_4
# %bb.3:
	jmp	.LBB222_6
.LBB222_4:
	jmp	.LBB222_5
.LBB222_5:
	callq	_invalid_parameter_noinfo_noreturn
.LBB222_6:
	jmp	.LBB222_7
.LBB222_7:
	movq	56(%rsp), %rcx
	movq	72(%rsp), %rax
	movq	%rcx, (%rax)
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
	.globl	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ" # -- Begin function ?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ
	.p2align	4, 0x90
"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ": # @"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
.seh_proc "?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 40(%rsp)
	callq	"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB223_1
	jmp	.LBB223_2
.LBB223_1:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	(%rax), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, 40(%rsp)
.LBB223_2:
	movq	40(%rsp), %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	.globl	"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ" # -- Begin function ?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ
	.p2align	4, 0x90
"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ": # @"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
.seh_proc "?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rcx, 80(%rsp)
	movq	80(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	%rax, 72(%rsp)
	movq	72(%rsp), %rcx
	callq	"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	movq	72(%rsp), %rcx
	callq	"?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB224_1
	jmp	.LBB224_2
.LBB224_1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 64(%rsp)
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 56(%rsp)
	movq	72(%rsp), %rcx
	callq	"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"
	movq	72(%rsp), %rcx
	callq	"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"
	movq	56(%rsp), %rcx
	movq	72(%rsp), %rax
	movq	24(%rax), %r8
	addq	$1, %r8
	movq	64(%rsp), %rdx
	callq	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
.LBB224_2:
	movq	72(%rsp), %rax
	movq	$0, 16(%rax)
	movq	72(%rsp), %rax
	movq	$15, 24(%rax)
	movb	$0, 55(%rsp)
	movq	72(%rsp), %rcx
	leaq	55(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	nop
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"
	.globl	"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z" # -- Begin function ??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z
	.p2align	4, 0x90
"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z": # @"??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"
.seh_proc "??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"
	.globl	"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ" # -- Begin function ?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ
	.p2align	4, 0x90
"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ": # @"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"
.seh_proc "?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z"
	.globl	"?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z" # -- Begin function ?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z
	.p2align	4, 0x90
"?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z": # @"?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z"
.seh_proc "?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	testb	$1, 8(%rax)
	jne	.LBB227_2
# %bb.1:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
.LBB227_2:
	movq	64(%rsp), %rax
	testb	$1, 8(%rax)
	jne	.LBB227_4
# %bb.3:
	movq	64(%rsp), %rcx
	callq	"?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
.LBB227_4:
	movq	48(%rsp), %rax                  # 8-byte Reload
	cmpq	$0, (%rax)
	jne	.LBB227_6
# %bb.5:
	movq	64(%rsp), %rcx
	movb	$1, %al
	cmpq	$0, (%rcx)
	movb	%al, 47(%rsp)                   # 1-byte Spill
	je	.LBB227_9
.LBB227_6:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	cmpq	$0, (%rcx)
	movb	%al, 46(%rsp)                   # 1-byte Spill
	je	.LBB227_8
# %bb.7:
	movq	64(%rsp), %rax
	cmpq	$0, (%rax)
	setne	%al
	movb	%al, 46(%rsp)                   # 1-byte Spill
.LBB227_8:
	movb	46(%rsp), %al                   # 1-byte Reload
	movb	%al, 47(%rsp)                   # 1-byte Spill
.LBB227_9:
	movb	47(%rsp), %al                   # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.globl	"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z" # -- Begin function ?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z
	.p2align	4, 0x90
"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z": # @"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
.Lfunc_begin39:
.seh_proc "?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$496, %rsp                      # imm = 0x1F0
	.seh_stackalloc 496
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 360(%rbp)
	movq	424(%rbp), %rax
	movq	416(%rbp), %rax
	movq	%r9, 344(%rbp)
	movq	%r8, 336(%rbp)
	movq	%rdx, 328(%rbp)
	movq	%rcx, 320(%rbp)
	movq	320(%rbp), %rax
	movq	%rax, 88(%rbp)                  # 8-byte Spill
	movq	416(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$12288, %eax                    # imm = 0x3000
	cmpl	$12288, %eax                    # imm = 0x3000
	jne	.LBB228_2
# %bb.1:
	movq	88(%rbp), %rcx                  # 8-byte Reload
	movq	424(%rbp), %rax
	movq	416(%rbp), %r10
	movq	344(%rbp), %r9
	movq	336(%rbp), %r8
	movq	328(%rbp), %rdx
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	movl	%eax, 356(%rbp)
	jmp	.LBB228_186
.LBB228_2:
	movq	416(%rbp), %rcx
	leaq	296(%rbp), %rdx
	movq	%rdx, 72(%rbp)                  # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	72(%rbp), %rcx                  # 8-byte Reload
.Ltmp200:
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Ltmp201:
	movq	%rax, 80(%rbp)                  # 8-byte Spill
	jmp	.LBB228_3
.LBB228_3:
	leaq	296(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	80(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 312(%rbp)
	movq	312(%rbp), %rcx
	leaq	264(%rbp), %rdx
	callq	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	movq	328(%rbp), %rax
	movq	%rax, 256(%rbp)
	movb	$0, 255(%rbp)
	movb	$0, 254(%rbp)
	movl	$10, 248(%rbp)
	movl	$12, 244(%rbp)
	movq	416(%rbp), %rcx
	leaq	200(%rbp), %rdx
	movq	%rdx, 56(%rbp)                  # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	56(%rbp), %rcx                  # 8-byte Reload
.Ltmp202:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp203:
	movq	%rax, 64(%rbp)                  # 8-byte Spill
	jmp	.LBB228_4
.LBB228_4:
	leaq	200(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	64(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 216(%rbp)
	movq	216(%rbp), %rax
	movq	%rax, 40(%rbp)                  # 8-byte Spill
	leaq	"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rcx
	movq	%rcx, 32(%rbp)                  # 8-byte Spill
	callq	"??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
	movq	32(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, 48(%rbp)                  # 8-byte Spill
	callq	"??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
	movq	40(%rbp), %rcx                  # 8-byte Reload
	movq	48(%rbp), %r8                   # 8-byte Reload
	movq	%rax, %rdx
.Ltmp204:
	leaq	229(%rbp), %r9
	callq	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
.Ltmp205:
	jmp	.LBB228_5
.LBB228_5:
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp206:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp207:
	movb	%al, 31(%rbp)                   # 1-byte Spill
	jmp	.LBB228_6
.LBB228_6:
	movb	31(%rbp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_7
	jmp	.LBB228_19
.LBB228_7:
	movq	336(%rbp), %rcx
.Ltmp208:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp209:
	movb	%al, 30(%rbp)                   # 1-byte Spill
	jmp	.LBB228_8
.LBB228_8:
	movb	30(%rbp), %al                   # 1-byte Reload
	movsbl	%al, %eax
	movsbl	240(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_13
# %bb.9:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 256(%rbp)
	movb	$43, (%rax)
	movq	336(%rbp), %rcx
.Ltmp214:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp215:
	jmp	.LBB228_10
.LBB228_10:
	jmp	.LBB228_18
.LBB228_13:
	movq	336(%rbp), %rcx
.Ltmp210:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp211:
	movb	%al, 29(%rbp)                   # 1-byte Spill
	jmp	.LBB228_14
.LBB228_14:
	movb	29(%rbp), %al                   # 1-byte Reload
	movsbl	%al, %eax
	movsbl	239(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_17
# %bb.15:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 256(%rbp)
	movb	$45, (%rax)
	movq	336(%rbp), %rcx
.Ltmp212:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp213:
	jmp	.LBB228_16
.LBB228_16:
	jmp	.LBB228_17
.LBB228_17:
	jmp	.LBB228_18
.LBB228_18:
	jmp	.LBB228_19
.LBB228_19:
	movq	256(%rbp), %rax
	movq	%rax, 192(%rbp)
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 256(%rbp)
	movb	$48, (%rax)
	movb	$0, 191(%rbp)
	movl	$0, 184(%rbp)
	movl	$0, 180(%rbp)
	movq	424(%rbp), %rdx
	movl	$36, %eax
	movl	$768, %ecx                      # imm = 0x300
	cmpl	$1000000000, (%rdx)             # imm = 0x3B9ACA00
	cmovel	%ecx, %eax
	movl	%eax, 164(%rbp)
	leaq	264(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	movq	%rax, 152(%rbp)
	movq	152(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	je	.LBB228_21
# %bb.20:
	movq	152(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$0, %eax
	jg	.LBB228_39
.LBB228_21:
	jmp	.LBB228_22
.LBB228_22:                             # =>This Inner Loop Header: Depth=1
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp232:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp233:
	movb	%al, 28(%rbp)                   # 1-byte Spill
	jmp	.LBB228_23
.LBB228_23:                             #   in Loop: Header=BB228_22 Depth=1
	movb	28(%rbp), %cl                   # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, 27(%rbp)                   # 1-byte Spill
	jne	.LBB228_24
	jmp	.LBB228_27
.LBB228_24:                             #   in Loop: Header=BB228_22 Depth=1
	movq	336(%rbp), %rcx
.Ltmp234:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp235:
	movb	%al, 26(%rbp)                   # 1-byte Spill
	jmp	.LBB228_25
.LBB228_25:                             #   in Loop: Header=BB228_22 Depth=1
.Ltmp236:
	movb	26(%rbp), %dl                   # 1-byte Reload
	leaq	229(%rbp), %rcx
	callq	"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
.Ltmp237:
	movq	%rax, 16(%rbp)                  # 8-byte Spill
	jmp	.LBB228_26
.LBB228_26:                             #   in Loop: Header=BB228_22 Depth=1
	movq	16(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 168(%rbp)
	cmpq	$10, %rax
	setb	%al
	movb	%al, 27(%rbp)                   # 1-byte Spill
.LBB228_27:                             #   in Loop: Header=BB228_22 Depth=1
	movb	27(%rbp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_28
	jmp	.LBB228_38
.LBB228_28:                             #   in Loop: Header=BB228_22 Depth=1
	movl	164(%rbp), %eax
	cmpl	184(%rbp), %eax
	jg	.LBB228_32
# %bb.29:                               #   in Loop: Header=BB228_22 Depth=1
	movl	180(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 180(%rbp)
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	168(%rbp), %rax
	jae	.LBB228_31
# %bb.30:                               #   in Loop: Header=BB228_22 Depth=1
	movb	$1, 254(%rbp)
.LBB228_31:                             #   in Loop: Header=BB228_22 Depth=1
	jmp	.LBB228_36
.LBB228_32:                             #   in Loop: Header=BB228_22 Depth=1
	cmpq	$0, 168(%rbp)
	jne	.LBB228_34
# %bb.33:                               #   in Loop: Header=BB228_22 Depth=1
	cmpl	$0, 184(%rbp)
	je	.LBB228_35
.LBB228_34:                             #   in Loop: Header=BB228_22 Depth=1
	movq	168(%rbp), %rcx
	leaq	"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	256(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 256(%rbp)
	movb	%cl, (%rax)
	movl	184(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 184(%rbp)
.LBB228_35:                             #   in Loop: Header=BB228_22 Depth=1
	jmp	.LBB228_36
.LBB228_36:                             #   in Loop: Header=BB228_22 Depth=1
	movb	$1, 191(%rbp)
	movq	336(%rbp), %rcx
.Ltmp296:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp297:
	jmp	.LBB228_37
.LBB228_37:                             #   in Loop: Header=BB228_22 Depth=1
	jmp	.LBB228_22
.LBB228_38:
	jmp	.LBB228_92
.LBB228_39:
	leaq	264(%rbp), %rcx
	callq	"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB228_40
	jmp	.LBB228_41
.LBB228_40:
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	movb	%al, 15(%rbp)                   # 1-byte Spill
	jmp	.LBB228_43
.LBB228_41:
	movq	312(%rbp), %rcx
.Ltmp216:
	callq	"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
.Ltmp217:
	movb	%al, 14(%rbp)                   # 1-byte Spill
	jmp	.LBB228_42
.LBB228_42:
	movb	14(%rbp), %al                   # 1-byte Reload
	movb	%al, 15(%rbp)                   # 1-byte Spill
	jmp	.LBB228_43
.LBB228_43:
	movb	15(%rbp), %al                   # 1-byte Reload
	movb	%al, 151(%rbp)
.Ltmp218:
	xorl	%eax, %eax
	movb	%al, %r8b
	leaq	112(%rbp), %rcx
	movl	$1, %edx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
.Ltmp219:
	jmp	.LBB228_44
.LBB228_44:
	movq	$0, 104(%rbp)
.LBB228_45:                             # =>This Inner Loop Header: Depth=1
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp220:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp221:
	movb	%al, 13(%rbp)                   # 1-byte Spill
	jmp	.LBB228_46
.LBB228_46:                             #   in Loop: Header=BB228_45 Depth=1
	movb	13(%rbp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_47
	jmp	.LBB228_70
.LBB228_47:                             #   in Loop: Header=BB228_45 Depth=1
	movq	336(%rbp), %rcx
.Ltmp222:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp223:
	movb	%al, 12(%rbp)                   # 1-byte Spill
	jmp	.LBB228_48
.LBB228_48:                             #   in Loop: Header=BB228_45 Depth=1
.Ltmp224:
	movb	12(%rbp), %dl                   # 1-byte Reload
	leaq	229(%rbp), %rcx
	callq	"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
.Ltmp225:
	movq	%rax, (%rbp)                    # 8-byte Spill
	jmp	.LBB228_49
.LBB228_49:                             #   in Loop: Header=BB228_45 Depth=1
	movq	(%rbp), %rax                    # 8-byte Reload
	movq	%rax, 168(%rbp)
	cmpq	$10, %rax
	jae	.LBB228_61
# %bb.50:                               #   in Loop: Header=BB228_45 Depth=1
	movb	$1, 191(%rbp)
	movl	164(%rbp), %eax
	cmpl	184(%rbp), %eax
	jg	.LBB228_54
# %bb.51:                               #   in Loop: Header=BB228_45 Depth=1
	movl	180(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 180(%rbp)
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	168(%rbp), %rax
	jae	.LBB228_53
# %bb.52:                               #   in Loop: Header=BB228_45 Depth=1
	movb	$1, 254(%rbp)
.LBB228_53:                             #   in Loop: Header=BB228_45 Depth=1
	jmp	.LBB228_58
.LBB228_54:                             #   in Loop: Header=BB228_45 Depth=1
	cmpq	$0, 168(%rbp)
	jne	.LBB228_56
# %bb.55:                               #   in Loop: Header=BB228_45 Depth=1
	cmpl	$0, 184(%rbp)
	je	.LBB228_57
.LBB228_56:                             #   in Loop: Header=BB228_45 Depth=1
	movq	168(%rbp), %rcx
	leaq	"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	256(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 256(%rbp)
	movb	%cl, (%rax)
	movl	184(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 184(%rbp)
.LBB228_57:                             #   in Loop: Header=BB228_45 Depth=1
	jmp	.LBB228_58
.LBB228_58:                             #   in Loop: Header=BB228_45 Depth=1
	movq	104(%rbp), %rdx
	leaq	112(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	je	.LBB228_60
# %bb.59:                               #   in Loop: Header=BB228_45 Depth=1
	movq	104(%rbp), %rdx
	leaq	112(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movb	(%rax), %cl
	addb	$1, %cl
	movb	%cl, (%rax)
.LBB228_60:                             #   in Loop: Header=BB228_45 Depth=1
	jmp	.LBB228_68
.LBB228_61:                             #   in Loop: Header=BB228_45 Depth=1
	movq	104(%rbp), %rdx
	leaq	112(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$0, %eax
	je	.LBB228_65
# %bb.62:                               #   in Loop: Header=BB228_45 Depth=1
	movsbl	151(%rbp), %eax
	cmpl	$0, %eax
	je	.LBB228_65
# %bb.63:                               #   in Loop: Header=BB228_45 Depth=1
	movq	336(%rbp), %rcx
.Ltmp226:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp227:
	movb	%al, -1(%rbp)                   # 1-byte Spill
	jmp	.LBB228_64
.LBB228_64:                             #   in Loop: Header=BB228_45 Depth=1
	movb	-1(%rbp), %al                   # 1-byte Reload
	movsbl	%al, %eax
	movsbl	151(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB228_66
.LBB228_65:
	jmp	.LBB228_70
.LBB228_66:                             #   in Loop: Header=BB228_45 Depth=1
.Ltmp228:
	xorl	%eax, %eax
	movb	%al, %dl
	leaq	112(%rbp), %rcx
	callq	"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
.Ltmp229:
	jmp	.LBB228_67
.LBB228_67:                             #   in Loop: Header=BB228_45 Depth=1
	movq	104(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 104(%rbp)
.LBB228_68:                             #   in Loop: Header=BB228_45 Depth=1
	movq	336(%rbp), %rcx
.Ltmp230:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp231:
	jmp	.LBB228_69
.LBB228_69:                             #   in Loop: Header=BB228_45 Depth=1
	jmp	.LBB228_45
.LBB228_70:
	cmpq	$0, 104(%rbp)
	je	.LBB228_75
# %bb.71:
	movq	104(%rbp), %rdx
	leaq	112(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB228_73
# %bb.72:
	movq	104(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 104(%rbp)
	jmp	.LBB228_74
.LBB228_73:
	movb	$1, 255(%rbp)
.LBB228_74:
	jmp	.LBB228_75
.LBB228_75:
	jmp	.LBB228_76
.LBB228_76:                             # =>This Inner Loop Header: Depth=1
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, 255(%rbp)
	movb	%al, -2(%rbp)                   # 1-byte Spill
	jne	.LBB228_78
# %bb.77:                               #   in Loop: Header=BB228_76 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	104(%rbp), %rax
	setb	%al
	movb	%al, -2(%rbp)                   # 1-byte Spill
.LBB228_78:                             #   in Loop: Header=BB228_76 Depth=1
	movb	-2(%rbp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_79
	jmp	.LBB228_90
.LBB228_79:                             #   in Loop: Header=BB228_76 Depth=1
	movq	152(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	jne	.LBB228_81
# %bb.80:
	jmp	.LBB228_90
.LBB228_81:                             #   in Loop: Header=BB228_76 Depth=1
	movq	104(%rbp), %rcx
	addq	$-1, %rcx
	movq	%rcx, 104(%rbp)
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	%rcx, %rax
	jae	.LBB228_83
# %bb.82:                               #   in Loop: Header=BB228_76 Depth=1
	movq	152(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -8(%rbp)                  # 4-byte Spill
	movq	104(%rbp), %rdx
	leaq	112(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	%rax, %rcx
	movl	-8(%rbp), %eax                  # 4-byte Reload
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_85
.LBB228_83:                             #   in Loop: Header=BB228_76 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	104(%rbp), %rax
	jne	.LBB228_86
# %bb.84:                               #   in Loop: Header=BB228_76 Depth=1
	movq	152(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -12(%rbp)                 # 4-byte Spill
	movq	104(%rbp), %rdx
	leaq	112(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	%rax, %rcx
	movl	-12(%rbp), %eax                 # 4-byte Reload
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jge	.LBB228_86
.LBB228_85:                             #   in Loop: Header=BB228_76 Depth=1
	movb	$1, 255(%rbp)
	jmp	.LBB228_89
.LBB228_86:                             #   in Loop: Header=BB228_76 Depth=1
	movq	152(%rbp), %rax
	movsbl	1(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB228_88
# %bb.87:                               #   in Loop: Header=BB228_76 Depth=1
	movq	152(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 152(%rbp)
.LBB228_88:                             #   in Loop: Header=BB228_76 Depth=1
	jmp	.LBB228_89
.LBB228_89:                             #   in Loop: Header=BB228_76 Depth=1
	jmp	.LBB228_76
.LBB228_90:
	leaq	112(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	jmp	.LBB228_92
.LBB228_92:
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp238:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp239:
	movb	%al, -13(%rbp)                  # 1-byte Spill
	jmp	.LBB228_93
.LBB228_93:
	movb	-13(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_94
	jmp	.LBB228_100
.LBB228_94:
	movq	336(%rbp), %rcx
.Ltmp240:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp241:
	movb	%al, -14(%rbp)                  # 1-byte Spill
	jmp	.LBB228_95
.LBB228_95:
	movb	-14(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movl	%eax, -20(%rbp)                 # 4-byte Spill
	movq	312(%rbp), %rcx
.Ltmp242:
	callq	"?decimal_point@?$numpunct@D@std@@QEBADXZ"
.Ltmp243:
	movb	%al, -15(%rbp)                  # 1-byte Spill
	jmp	.LBB228_96
.LBB228_96:
	movl	-20(%rbp), %eax                 # 4-byte Reload
	movb	-15(%rbp), %cl                  # 1-byte Reload
	movsbl	%cl, %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_100
# %bb.97:
.Ltmp244:
	callq	localeconv
.Ltmp245:
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	jmp	.LBB228_98
.LBB228_98:
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	movb	(%rax), %cl
	movq	256(%rbp), %rax
	movq	%rax, %rdx
	incq	%rdx
	movq	%rdx, 256(%rbp)
	movb	%cl, (%rax)
	movq	336(%rbp), %rcx
.Ltmp246:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp247:
	jmp	.LBB228_99
.LBB228_99:
	jmp	.LBB228_100
.LBB228_100:
	movq	424(%rbp), %rax
	cmpl	$1000000000, (%rax)             # imm = 0x3B9ACA00
	je	.LBB228_113
# %bb.101:
	cmpl	$0, 184(%rbp)
	jne	.LBB228_113
# %bb.102:
	jmp	.LBB228_103
.LBB228_103:                            # =>This Inner Loop Header: Depth=1
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp248:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp249:
	movb	%al, -33(%rbp)                  # 1-byte Spill
	jmp	.LBB228_104
.LBB228_104:                            #   in Loop: Header=BB228_103 Depth=1
	movb	-33(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -34(%rbp)                  # 1-byte Spill
	jne	.LBB228_105
	jmp	.LBB228_107
.LBB228_105:                            #   in Loop: Header=BB228_103 Depth=1
	movq	336(%rbp), %rcx
.Ltmp250:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp251:
	movb	%al, -35(%rbp)                  # 1-byte Spill
	jmp	.LBB228_106
.LBB228_106:                            #   in Loop: Header=BB228_103 Depth=1
	movb	-35(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	229(%rbp), %ecx
	cmpl	%ecx, %eax
	sete	%al
	movb	%al, -34(%rbp)                  # 1-byte Spill
.LBB228_107:                            #   in Loop: Header=BB228_103 Depth=1
	movb	-34(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_108
	jmp	.LBB228_110
.LBB228_108:                            #   in Loop: Header=BB228_103 Depth=1
	movl	180(%rbp), %eax
	decl	%eax
	movl	%eax, 180(%rbp)
	movb	$1, 191(%rbp)
	movq	336(%rbp), %rcx
.Ltmp294:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp295:
	jmp	.LBB228_109
.LBB228_109:                            #   in Loop: Header=BB228_103 Depth=1
	jmp	.LBB228_103
.LBB228_110:
	cmpl	$0, 180(%rbp)
	jge	.LBB228_112
# %bb.111:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 256(%rbp)
	movb	$48, (%rax)
	movl	180(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 180(%rbp)
.LBB228_112:
	jmp	.LBB228_113
.LBB228_113:
	jmp	.LBB228_114
.LBB228_114:                            # =>This Inner Loop Header: Depth=1
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp252:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp253:
	movb	%al, -36(%rbp)                  # 1-byte Spill
	jmp	.LBB228_115
.LBB228_115:                            #   in Loop: Header=BB228_114 Depth=1
	movb	-36(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -37(%rbp)                  # 1-byte Spill
	jne	.LBB228_116
	jmp	.LBB228_119
.LBB228_116:                            #   in Loop: Header=BB228_114 Depth=1
	movq	336(%rbp), %rcx
.Ltmp254:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp255:
	movb	%al, -38(%rbp)                  # 1-byte Spill
	jmp	.LBB228_117
.LBB228_117:                            #   in Loop: Header=BB228_114 Depth=1
.Ltmp256:
	movb	-38(%rbp), %dl                  # 1-byte Reload
	leaq	229(%rbp), %rcx
	callq	"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
.Ltmp257:
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	jmp	.LBB228_118
.LBB228_118:                            #   in Loop: Header=BB228_114 Depth=1
	movq	-48(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 168(%rbp)
	cmpq	$10, %rax
	setb	%al
	movb	%al, -37(%rbp)                  # 1-byte Spill
.LBB228_119:                            #   in Loop: Header=BB228_114 Depth=1
	movb	-37(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_120
	jmp	.LBB228_127
.LBB228_120:                            #   in Loop: Header=BB228_114 Depth=1
	movl	184(%rbp), %eax
	cmpl	164(%rbp), %eax
	jge	.LBB228_122
# %bb.121:                              #   in Loop: Header=BB228_114 Depth=1
	movq	168(%rbp), %rcx
	leaq	"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	256(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 256(%rbp)
	movb	%cl, (%rax)
	movl	184(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 184(%rbp)
	jmp	.LBB228_125
.LBB228_122:                            #   in Loop: Header=BB228_114 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	168(%rbp), %rax
	jae	.LBB228_124
# %bb.123:                              #   in Loop: Header=BB228_114 Depth=1
	movb	$1, 254(%rbp)
.LBB228_124:                            #   in Loop: Header=BB228_114 Depth=1
	jmp	.LBB228_125
.LBB228_125:                            #   in Loop: Header=BB228_114 Depth=1
	movb	$1, 191(%rbp)
	movq	336(%rbp), %rcx
.Ltmp292:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp293:
	jmp	.LBB228_126
.LBB228_126:                            #   in Loop: Header=BB228_114 Depth=1
	jmp	.LBB228_114
.LBB228_127:
	testb	$1, 254(%rbp)
	je	.LBB228_139
# %bb.128:
	movq	256(%rbp), %rax
	movq	%rax, 96(%rbp)
.LBB228_129:                            # =>This Inner Loop Header: Depth=1
	movq	96(%rbp), %rax
	addq	$-1, %rax
	movq	%rax, 96(%rbp)
	cmpq	192(%rbp), %rax
	je	.LBB228_136
# %bb.130:                              #   in Loop: Header=BB228_129 Depth=1
	movq	96(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -60(%rbp)                 # 4-byte Spill
.Ltmp258:
	callq	localeconv
.Ltmp259:
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	jmp	.LBB228_131
.LBB228_131:                            #   in Loop: Header=BB228_129 Depth=1
	movl	-60(%rbp), %eax                 # 4-byte Reload
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rcx
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	je	.LBB228_135
# %bb.132:                              #   in Loop: Header=BB228_129 Depth=1
	movq	96(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$57, %eax
	je	.LBB228_134
# %bb.133:
	movq	96(%rbp), %rax
	movb	(%rax), %cl
	addb	$1, %cl
	movb	%cl, (%rax)
	jmp	.LBB228_136
.LBB228_134:                            #   in Loop: Header=BB228_129 Depth=1
	movq	96(%rbp), %rax
	movb	$48, (%rax)
.LBB228_135:                            #   in Loop: Header=BB228_129 Depth=1
	jmp	.LBB228_129
.LBB228_136:
	movq	96(%rbp), %rax
	cmpq	192(%rbp), %rax
	jne	.LBB228_138
# %bb.137:
	movq	96(%rbp), %rax
	movb	$49, (%rax)
	movl	180(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 180(%rbp)
.LBB228_138:
	jmp	.LBB228_139
.LBB228_139:
	testb	$1, 191(%rbp)
	je	.LBB228_181
# %bb.140:
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp260:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp261:
	movb	%al, -61(%rbp)                  # 1-byte Spill
	jmp	.LBB228_141
.LBB228_141:
	movb	-61(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_142
	jmp	.LBB228_181
.LBB228_142:
	movq	336(%rbp), %rcx
.Ltmp262:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp263:
	movb	%al, -62(%rbp)                  # 1-byte Spill
	jmp	.LBB228_143
.LBB228_143:
	movb	-62(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	242(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB228_146
# %bb.144:
	movq	336(%rbp), %rcx
.Ltmp264:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp265:
	movb	%al, -63(%rbp)                  # 1-byte Spill
	jmp	.LBB228_145
.LBB228_145:
	movb	-63(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	241(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_181
.LBB228_146:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 256(%rbp)
	movb	$101, (%rax)
	movq	336(%rbp), %rcx
.Ltmp266:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp267:
	jmp	.LBB228_147
.LBB228_147:
	movb	$0, 191(%rbp)
	movl	$0, 184(%rbp)
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp268:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp269:
	movb	%al, -64(%rbp)                  # 1-byte Spill
	jmp	.LBB228_148
.LBB228_148:
	movb	-64(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_149
	jmp	.LBB228_159
.LBB228_149:
	movq	336(%rbp), %rcx
.Ltmp270:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp271:
	movb	%al, -65(%rbp)                  # 1-byte Spill
	jmp	.LBB228_150
.LBB228_150:
	movb	-65(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	240(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_153
# %bb.151:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 256(%rbp)
	movb	$43, (%rax)
	movq	336(%rbp), %rcx
.Ltmp276:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp277:
	jmp	.LBB228_152
.LBB228_152:
	jmp	.LBB228_158
.LBB228_153:
	movq	336(%rbp), %rcx
.Ltmp272:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp273:
	movb	%al, -66(%rbp)                  # 1-byte Spill
	jmp	.LBB228_154
.LBB228_154:
	movb	-66(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	239(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB228_157
# %bb.155:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 256(%rbp)
	movb	$45, (%rax)
	movq	336(%rbp), %rcx
.Ltmp274:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp275:
	jmp	.LBB228_156
.LBB228_156:
	jmp	.LBB228_157
.LBB228_157:
	jmp	.LBB228_158
.LBB228_158:
	jmp	.LBB228_159
.LBB228_159:
	jmp	.LBB228_160
.LBB228_160:                            # =>This Inner Loop Header: Depth=1
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp278:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp279:
	movb	%al, -67(%rbp)                  # 1-byte Spill
	jmp	.LBB228_161
.LBB228_161:                            #   in Loop: Header=BB228_160 Depth=1
	movb	-67(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -68(%rbp)                  # 1-byte Spill
	jne	.LBB228_162
	jmp	.LBB228_164
.LBB228_162:                            #   in Loop: Header=BB228_160 Depth=1
	movq	336(%rbp), %rcx
.Ltmp280:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp281:
	movb	%al, -69(%rbp)                  # 1-byte Spill
	jmp	.LBB228_163
.LBB228_163:                            #   in Loop: Header=BB228_160 Depth=1
	movb	-69(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	229(%rbp), %ecx
	cmpl	%ecx, %eax
	sete	%al
	movb	%al, -68(%rbp)                  # 1-byte Spill
.LBB228_164:                            #   in Loop: Header=BB228_160 Depth=1
	movb	-68(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_165
	jmp	.LBB228_167
.LBB228_165:                            #   in Loop: Header=BB228_160 Depth=1
	movb	$1, 191(%rbp)
	movq	336(%rbp), %rcx
.Ltmp290:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp291:
	jmp	.LBB228_166
.LBB228_166:                            #   in Loop: Header=BB228_160 Depth=1
	jmp	.LBB228_160
.LBB228_167:
	testb	$1, 191(%rbp)
	je	.LBB228_169
# %bb.168:
	movq	256(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 256(%rbp)
	movb	$48, (%rax)
.LBB228_169:
	jmp	.LBB228_170
.LBB228_170:                            # =>This Inner Loop Header: Depth=1
	movq	344(%rbp), %rdx
	movq	336(%rbp), %rcx
.Ltmp282:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp283:
	movb	%al, -70(%rbp)                  # 1-byte Spill
	jmp	.LBB228_171
.LBB228_171:                            #   in Loop: Header=BB228_170 Depth=1
	movb	-70(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -71(%rbp)                  # 1-byte Spill
	jne	.LBB228_172
	jmp	.LBB228_175
.LBB228_172:                            #   in Loop: Header=BB228_170 Depth=1
	movq	336(%rbp), %rcx
.Ltmp284:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp285:
	movb	%al, -72(%rbp)                  # 1-byte Spill
	jmp	.LBB228_173
.LBB228_173:                            #   in Loop: Header=BB228_170 Depth=1
.Ltmp286:
	movb	-72(%rbp), %dl                  # 1-byte Reload
	leaq	229(%rbp), %rcx
	callq	"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
.Ltmp287:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB228_174
.LBB228_174:                            #   in Loop: Header=BB228_170 Depth=1
	movq	-80(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 168(%rbp)
	cmpq	$10, %rax
	setb	%al
	movb	%al, -71(%rbp)                  # 1-byte Spill
.LBB228_175:                            #   in Loop: Header=BB228_170 Depth=1
	movb	-71(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB228_176
	jmp	.LBB228_180
.LBB228_176:                            #   in Loop: Header=BB228_170 Depth=1
	cmpl	$8, 184(%rbp)
	jge	.LBB228_178
# %bb.177:                              #   in Loop: Header=BB228_170 Depth=1
	movq	168(%rbp), %rcx
	leaq	"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	256(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 256(%rbp)
	movb	%cl, (%rax)
	movl	184(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 184(%rbp)
.LBB228_178:                            #   in Loop: Header=BB228_170 Depth=1
	movb	$1, 191(%rbp)
	movq	336(%rbp), %rcx
.Ltmp288:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp289:
	jmp	.LBB228_179
.LBB228_179:                            #   in Loop: Header=BB228_170 Depth=1
	jmp	.LBB228_170
.LBB228_180:
	jmp	.LBB228_181
.LBB228_181:
	testb	$1, 255(%rbp)
	jne	.LBB228_183
# %bb.182:
	testb	$1, 191(%rbp)
	jne	.LBB228_184
.LBB228_183:
	movq	328(%rbp), %rax
	movq	%rax, 256(%rbp)
.LBB228_184:
	movq	256(%rbp), %rax
	movb	$0, (%rax)
	movl	180(%rbp), %eax
	movl	%eax, 356(%rbp)
	leaq	264(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	jmp	.LBB228_186
.LBB228_186:
	movl	356(%rbp), %eax
	addq	$496, %rsp                      # imm = 0x1F0
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$11@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$11@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$11@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB228_11:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	296(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$12@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$12@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$12@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB228_12:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	200(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$91@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$91@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$91@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB228_91:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	112(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$185@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$185@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$185@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB228_185:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	264(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end39:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.p2align	2
"$cppxdata$?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z":
	.long	429065506                       # MagicNumber
	.long	4                               # MaxState
	.long	("$stateUnwindMap$?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	8                               # IPMapEntries
	.long	("$ip2state$?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z")@IMGREL # IPToStateXData
	.long	488                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z":
	.long	-1                              # ToState
	.long	"?dtor$11@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$185@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$91@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$12@?0??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
"$ip2state$?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z":
	.long	.Lfunc_begin39@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp200@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp201@IMGREL+1               # IP
	.long	-1                              # ToState
	.long	.Ltmp202@IMGREL+1               # IP
	.long	3                               # ToState
	.long	.Ltmp204@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp220@IMGREL+1               # IP
	.long	2                               # ToState
	.long	.Ltmp238@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp289@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
                                        # -- End function
	.def	"?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	__real@4024000000000000         # -- Begin function ?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z
	.section	.rdata,"dr",discard,__real@4024000000000000
	.p2align	3
__real@4024000000000000:
	.quad	0x4024000000000000              # double 10
	.section	.text,"xr",discard,"?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z"
	.globl	"?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z"
	.p2align	4, 0x90
"?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z": # @"?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z"
.seh_proc "?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r9, 80(%rsp)
	movl	%r8d, 76(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	callq	_errno
	movq	%rax, 48(%rsp)
	movq	48(%rsp), %rax
	movl	(%rax), %eax
	movl	%eax, 44(%rsp)
	movq	48(%rsp), %rax
	movl	$0, (%rax)
	movq	64(%rsp), %rdx
	movq	56(%rsp), %rcx
	callq	strtod
	movsd	%xmm0, 32(%rsp)
	movq	48(%rsp), %rax
	movl	(%rax), %ecx
	movq	80(%rsp), %rax
	movl	%ecx, (%rax)
	movl	44(%rsp), %ecx
	movq	48(%rsp), %rax
	movl	%ecx, (%rax)
	cmpl	$0, 76(%rsp)
	je	.LBB229_2
# %bb.1:
	cvtsi2sdl	76(%rsp), %xmm1
	movsd	__real@4024000000000000(%rip), %xmm0 # xmm0 = mem[0],zero
	callq	pow
	mulsd	32(%rsp), %xmm0
	movsd	%xmm0, 32(%rsp)
.LBB229_2:
	movsd	32(%rsp), %xmm0                 # xmm0 = mem[0],zero
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.globl	"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z" # -- Begin function ?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z
	.p2align	4, 0x90
"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z": # @"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
.Lfunc_begin40:
.seh_proc "?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$480, %rsp                      # imm = 0x1E0
	.seh_stackalloc 480
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 344(%rbp)
	movq	408(%rbp), %rax
	movq	400(%rbp), %rax
	movq	%r9, 336(%rbp)
	movq	%r8, 328(%rbp)
	movq	%rdx, 320(%rbp)
	movq	%rcx, 312(%rbp)
	movq	400(%rbp), %rcx
	leaq	288(%rbp), %rdx
	movq	%rdx, 72(%rbp)                  # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	72(%rbp), %rcx                  # 8-byte Reload
.Ltmp298:
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Ltmp299:
	movq	%rax, 80(%rbp)                  # 8-byte Spill
	jmp	.LBB230_1
.LBB230_1:
	leaq	288(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	80(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 304(%rbp)
	movq	304(%rbp), %rcx
	leaq	256(%rbp), %rdx
	callq	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	movl	$22, 252(%rbp)
	movl	$24, 248(%rbp)
	movl	$26, 244(%rbp)
	movq	400(%rbp), %rcx
	leaq	184(%rbp), %rdx
	movq	%rdx, 56(%rbp)                  # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	56(%rbp), %rcx                  # 8-byte Reload
.Ltmp300:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp301:
	movq	%rax, 64(%rbp)                  # 8-byte Spill
	jmp	.LBB230_2
.LBB230_2:
	leaq	184(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	64(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 200(%rbp)
	movq	200(%rbp), %rax
	movq	%rax, 40(%rbp)                  # 8-byte Spill
	leaq	"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rcx
	movq	%rcx, 32(%rbp)                  # 8-byte Spill
	callq	"??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
	movq	32(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, 48(%rbp)                  # 8-byte Spill
	callq	"??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
	movq	40(%rbp), %rcx                  # 8-byte Reload
	movq	48(%rbp), %r8                   # 8-byte Reload
	movq	%rax, %rdx
.Ltmp302:
	leaq	208(%rbp), %r9
	callq	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
.Ltmp303:
	jmp	.LBB230_3
.LBB230_3:
	movq	320(%rbp), %rax
	movq	%rax, 176(%rbp)
	movb	$0, 175(%rbp)
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp304:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp305:
	movb	%al, 31(%rbp)                   # 1-byte Spill
	jmp	.LBB230_4
.LBB230_4:
	movb	31(%rbp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_5
	jmp	.LBB230_17
.LBB230_5:
	movq	328(%rbp), %rcx
.Ltmp306:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp307:
	movb	%al, 30(%rbp)                   # 1-byte Spill
	jmp	.LBB230_6
.LBB230_6:
	movb	30(%rbp), %al                   # 1-byte Reload
	movsbl	%al, %eax
	movsbl	231(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_11
# %bb.7:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$43, (%rax)
	movq	328(%rbp), %rcx
.Ltmp312:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp313:
	jmp	.LBB230_8
.LBB230_8:
	jmp	.LBB230_16
.LBB230_11:
	movq	328(%rbp), %rcx
.Ltmp308:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp309:
	movb	%al, 29(%rbp)                   # 1-byte Spill
	jmp	.LBB230_12
.LBB230_12:
	movb	29(%rbp), %al                   # 1-byte Reload
	movsbl	%al, %eax
	movsbl	230(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_15
# %bb.13:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$45, (%rax)
	movq	328(%rbp), %rcx
.Ltmp310:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp311:
	jmp	.LBB230_14
.LBB230_14:
	jmp	.LBB230_15
.LBB230_15:
	jmp	.LBB230_16
.LBB230_16:
	jmp	.LBB230_17
.LBB230_17:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$48, (%rax)
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$120, (%rax)
	movb	$0, 159(%rbp)
	movl	$0, 152(%rbp)
	movl	$0, 148(%rbp)
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp314:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp315:
	movb	%al, 28(%rbp)                   # 1-byte Spill
	jmp	.LBB230_18
.LBB230_18:
	movb	28(%rbp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_19
	jmp	.LBB230_32
.LBB230_19:
	movq	328(%rbp), %rcx
.Ltmp316:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp317:
	movb	%al, 27(%rbp)                   # 1-byte Spill
	jmp	.LBB230_20
.LBB230_20:
	movb	27(%rbp), %al                   # 1-byte Reload
	movsbl	%al, %eax
	movsbl	208(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_32
# %bb.21:
	movq	336(%rbp), %rax
	movq	%rax, 8(%rbp)                   # 8-byte Spill
	movq	328(%rbp), %rcx
.Ltmp318:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp319:
	movq	%rax, 16(%rbp)                  # 8-byte Spill
	jmp	.LBB230_22
.LBB230_22:
.Ltmp320:
	movq	8(%rbp), %rdx                   # 8-byte Reload
	movq	16(%rbp), %rcx                  # 8-byte Reload
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp321:
	movb	%al, 7(%rbp)                    # 1-byte Spill
	jmp	.LBB230_23
.LBB230_23:
	movb	7(%rbp), %al                    # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_24
	jmp	.LBB230_30
.LBB230_24:
	movq	328(%rbp), %rcx
.Ltmp322:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp323:
	movb	%al, 6(%rbp)                    # 1-byte Spill
	jmp	.LBB230_25
.LBB230_25:
	movb	6(%rbp), %al                    # 1-byte Reload
	movsbl	%al, %eax
	movsbl	233(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB230_28
# %bb.26:
	movq	328(%rbp), %rcx
.Ltmp324:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp325:
	movb	%al, 5(%rbp)                    # 1-byte Spill
	jmp	.LBB230_27
.LBB230_27:
	movb	5(%rbp), %al                    # 1-byte Reload
	movsbl	%al, %eax
	movsbl	232(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_30
.LBB230_28:
	movq	328(%rbp), %rcx
.Ltmp326:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp327:
	jmp	.LBB230_29
.LBB230_29:
	jmp	.LBB230_31
.LBB230_30:
	movb	$1, 159(%rbp)
.LBB230_31:
	jmp	.LBB230_32
.LBB230_32:
	movq	408(%rbp), %rdx
	movl	$36, %eax
	movl	$768, %ecx                      # imm = 0x300
	cmpl	$1000000000, (%rdx)             # imm = 0x3B9ACA00
	cmovel	%ecx, %eax
	movl	%eax, 144(%rbp)
	leaq	256(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	movq	%rax, 136(%rbp)
	movq	136(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	je	.LBB230_34
# %bb.33:
	movq	136(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$0, %eax
	jg	.LBB230_50
.LBB230_34:
	jmp	.LBB230_35
.LBB230_35:                             # =>This Inner Loop Header: Depth=1
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp344:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp345:
	movb	%al, 4(%rbp)                    # 1-byte Spill
	jmp	.LBB230_36
.LBB230_36:                             #   in Loop: Header=BB230_35 Depth=1
	movb	4(%rbp), %cl                    # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, 3(%rbp)                    # 1-byte Spill
	jne	.LBB230_37
	jmp	.LBB230_40
.LBB230_37:                             #   in Loop: Header=BB230_35 Depth=1
	movq	328(%rbp), %rcx
.Ltmp346:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp347:
	movb	%al, 2(%rbp)                    # 1-byte Spill
	jmp	.LBB230_38
.LBB230_38:                             #   in Loop: Header=BB230_35 Depth=1
.Ltmp348:
	movb	2(%rbp), %dl                    # 1-byte Reload
	leaq	208(%rbp), %rcx
	callq	"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
.Ltmp349:
	movq	%rax, -8(%rbp)                  # 8-byte Spill
	jmp	.LBB230_39
.LBB230_39:                             #   in Loop: Header=BB230_35 Depth=1
	movq	-8(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 160(%rbp)
	cmpq	$22, %rax
	setb	%al
	movb	%al, 3(%rbp)                    # 1-byte Spill
.LBB230_40:                             #   in Loop: Header=BB230_35 Depth=1
	movb	3(%rbp), %al                    # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_41
	jmp	.LBB230_49
.LBB230_41:                             #   in Loop: Header=BB230_35 Depth=1
	movl	144(%rbp), %eax
	cmpl	152(%rbp), %eax
	jg	.LBB230_43
# %bb.42:                               #   in Loop: Header=BB230_35 Depth=1
	movl	148(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 148(%rbp)
	jmp	.LBB230_47
.LBB230_43:                             #   in Loop: Header=BB230_35 Depth=1
	cmpq	$0, 160(%rbp)
	jne	.LBB230_45
# %bb.44:                               #   in Loop: Header=BB230_35 Depth=1
	cmpl	$0, 152(%rbp)
	je	.LBB230_46
.LBB230_45:                             #   in Loop: Header=BB230_35 Depth=1
	movq	160(%rbp), %rcx
	leaq	"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	176(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 176(%rbp)
	movb	%cl, (%rax)
	movl	152(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 152(%rbp)
.LBB230_46:                             #   in Loop: Header=BB230_35 Depth=1
	jmp	.LBB230_47
.LBB230_47:                             #   in Loop: Header=BB230_35 Depth=1
	movb	$1, 159(%rbp)
	movq	328(%rbp), %rcx
.Ltmp406:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp407:
	jmp	.LBB230_48
.LBB230_48:                             #   in Loop: Header=BB230_35 Depth=1
	jmp	.LBB230_35
.LBB230_49:
	jmp	.LBB230_101
.LBB230_50:
	leaq	256(%rbp), %rcx
	callq	"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB230_51
	jmp	.LBB230_52
.LBB230_51:
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	movb	%al, -9(%rbp)                   # 1-byte Spill
	jmp	.LBB230_54
.LBB230_52:
	movq	304(%rbp), %rcx
.Ltmp328:
	callq	"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
.Ltmp329:
	movb	%al, -10(%rbp)                  # 1-byte Spill
	jmp	.LBB230_53
.LBB230_53:
	movb	-10(%rbp), %al                  # 1-byte Reload
	movb	%al, -9(%rbp)                   # 1-byte Spill
	jmp	.LBB230_54
.LBB230_54:
	movb	-9(%rbp), %al                   # 1-byte Reload
	movb	%al, 135(%rbp)
.Ltmp330:
	xorl	%eax, %eax
	movb	%al, %r8b
	leaq	96(%rbp), %rcx
	movl	$1, %edx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
.Ltmp331:
	jmp	.LBB230_55
.LBB230_55:
	movq	$0, 88(%rbp)
.LBB230_56:                             # =>This Inner Loop Header: Depth=1
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp332:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp333:
	movb	%al, -11(%rbp)                  # 1-byte Spill
	jmp	.LBB230_57
.LBB230_57:                             #   in Loop: Header=BB230_56 Depth=1
	movb	-11(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_58
	jmp	.LBB230_79
.LBB230_58:                             #   in Loop: Header=BB230_56 Depth=1
	movq	328(%rbp), %rcx
.Ltmp334:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp335:
	movb	%al, -12(%rbp)                  # 1-byte Spill
	jmp	.LBB230_59
.LBB230_59:                             #   in Loop: Header=BB230_56 Depth=1
.Ltmp336:
	movb	-12(%rbp), %dl                  # 1-byte Reload
	leaq	208(%rbp), %rcx
	callq	"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
.Ltmp337:
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	jmp	.LBB230_60
.LBB230_60:                             #   in Loop: Header=BB230_56 Depth=1
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 160(%rbp)
	cmpq	$22, %rax
	jae	.LBB230_70
# %bb.61:                               #   in Loop: Header=BB230_56 Depth=1
	movb	$1, 159(%rbp)
	movl	144(%rbp), %eax
	cmpl	152(%rbp), %eax
	jg	.LBB230_63
# %bb.62:                               #   in Loop: Header=BB230_56 Depth=1
	movl	148(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 148(%rbp)
	jmp	.LBB230_67
.LBB230_63:                             #   in Loop: Header=BB230_56 Depth=1
	cmpq	$0, 160(%rbp)
	jne	.LBB230_65
# %bb.64:                               #   in Loop: Header=BB230_56 Depth=1
	cmpl	$0, 152(%rbp)
	je	.LBB230_66
.LBB230_65:                             #   in Loop: Header=BB230_56 Depth=1
	movq	160(%rbp), %rcx
	leaq	"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	176(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 176(%rbp)
	movb	%cl, (%rax)
	movl	152(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 152(%rbp)
.LBB230_66:                             #   in Loop: Header=BB230_56 Depth=1
	jmp	.LBB230_67
.LBB230_67:                             #   in Loop: Header=BB230_56 Depth=1
	movq	88(%rbp), %rdx
	leaq	96(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	je	.LBB230_69
# %bb.68:                               #   in Loop: Header=BB230_56 Depth=1
	movq	88(%rbp), %rdx
	leaq	96(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movb	(%rax), %cl
	addb	$1, %cl
	movb	%cl, (%rax)
.LBB230_69:                             #   in Loop: Header=BB230_56 Depth=1
	jmp	.LBB230_77
.LBB230_70:                             #   in Loop: Header=BB230_56 Depth=1
	movq	88(%rbp), %rdx
	leaq	96(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$0, %eax
	je	.LBB230_74
# %bb.71:                               #   in Loop: Header=BB230_56 Depth=1
	movsbl	135(%rbp), %eax
	cmpl	$0, %eax
	je	.LBB230_74
# %bb.72:                               #   in Loop: Header=BB230_56 Depth=1
	movq	328(%rbp), %rcx
.Ltmp338:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp339:
	movb	%al, -25(%rbp)                  # 1-byte Spill
	jmp	.LBB230_73
.LBB230_73:                             #   in Loop: Header=BB230_56 Depth=1
	movb	-25(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	135(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB230_75
.LBB230_74:
	jmp	.LBB230_79
.LBB230_75:                             #   in Loop: Header=BB230_56 Depth=1
.Ltmp340:
	xorl	%eax, %eax
	movb	%al, %dl
	leaq	96(%rbp), %rcx
	callq	"?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
.Ltmp341:
	jmp	.LBB230_76
.LBB230_76:                             #   in Loop: Header=BB230_56 Depth=1
	movq	88(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 88(%rbp)
.LBB230_77:                             #   in Loop: Header=BB230_56 Depth=1
	movq	328(%rbp), %rcx
.Ltmp342:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp343:
	jmp	.LBB230_78
.LBB230_78:                             #   in Loop: Header=BB230_56 Depth=1
	jmp	.LBB230_56
.LBB230_79:
	cmpq	$0, 88(%rbp)
	je	.LBB230_84
# %bb.80:
	movq	88(%rbp), %rdx
	leaq	96(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB230_82
# %bb.81:
	movq	88(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 88(%rbp)
	jmp	.LBB230_83
.LBB230_82:
	movb	$1, 175(%rbp)
.LBB230_83:
	jmp	.LBB230_84
.LBB230_84:
	jmp	.LBB230_85
.LBB230_85:                             # =>This Inner Loop Header: Depth=1
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, 175(%rbp)
	movb	%al, -26(%rbp)                  # 1-byte Spill
	jne	.LBB230_87
# %bb.86:                               #   in Loop: Header=BB230_85 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	88(%rbp), %rax
	setb	%al
	movb	%al, -26(%rbp)                  # 1-byte Spill
.LBB230_87:                             #   in Loop: Header=BB230_85 Depth=1
	movb	-26(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_88
	jmp	.LBB230_99
.LBB230_88:                             #   in Loop: Header=BB230_85 Depth=1
	movq	136(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	jne	.LBB230_90
# %bb.89:
	jmp	.LBB230_99
.LBB230_90:                             #   in Loop: Header=BB230_85 Depth=1
	movq	88(%rbp), %rcx
	addq	$-1, %rcx
	movq	%rcx, 88(%rbp)
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	%rcx, %rax
	jae	.LBB230_92
# %bb.91:                               #   in Loop: Header=BB230_85 Depth=1
	movq	136(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -32(%rbp)                 # 4-byte Spill
	movq	88(%rbp), %rdx
	leaq	96(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	%rax, %rcx
	movl	-32(%rbp), %eax                 # 4-byte Reload
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_94
.LBB230_92:                             #   in Loop: Header=BB230_85 Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	88(%rbp), %rax
	jne	.LBB230_95
# %bb.93:                               #   in Loop: Header=BB230_85 Depth=1
	movq	136(%rbp), %rax
	movsbl	(%rax), %eax
	movl	%eax, -36(%rbp)                 # 4-byte Spill
	movq	88(%rbp), %rdx
	leaq	96(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	%rax, %rcx
	movl	-36(%rbp), %eax                 # 4-byte Reload
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jge	.LBB230_95
.LBB230_94:                             #   in Loop: Header=BB230_85 Depth=1
	movb	$1, 175(%rbp)
	jmp	.LBB230_98
.LBB230_95:                             #   in Loop: Header=BB230_85 Depth=1
	movq	136(%rbp), %rax
	movsbl	1(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB230_97
# %bb.96:                               #   in Loop: Header=BB230_85 Depth=1
	movq	136(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 136(%rbp)
.LBB230_97:                             #   in Loop: Header=BB230_85 Depth=1
	jmp	.LBB230_98
.LBB230_98:                             #   in Loop: Header=BB230_85 Depth=1
	jmp	.LBB230_85
.LBB230_99:
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	jmp	.LBB230_101
.LBB230_101:
	testb	$1, 159(%rbp)
	je	.LBB230_104
# %bb.102:
	cmpl	$0, 152(%rbp)
	jne	.LBB230_104
# %bb.103:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 176(%rbp)
	movb	$48, (%rax)
.LBB230_104:
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp350:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp351:
	movb	%al, -37(%rbp)                  # 1-byte Spill
	jmp	.LBB230_105
.LBB230_105:
	movb	-37(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_106
	jmp	.LBB230_112
.LBB230_106:
	movq	328(%rbp), %rcx
.Ltmp352:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp353:
	movb	%al, -38(%rbp)                  # 1-byte Spill
	jmp	.LBB230_107
.LBB230_107:
	movb	-38(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movl	%eax, -44(%rbp)                 # 4-byte Spill
	movq	304(%rbp), %rcx
.Ltmp354:
	callq	"?decimal_point@?$numpunct@D@std@@QEBADXZ"
.Ltmp355:
	movb	%al, -39(%rbp)                  # 1-byte Spill
	jmp	.LBB230_108
.LBB230_108:
	movl	-44(%rbp), %eax                 # 4-byte Reload
	movb	-39(%rbp), %cl                  # 1-byte Reload
	movsbl	%cl, %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_112
# %bb.109:
.Ltmp356:
	callq	localeconv
.Ltmp357:
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	jmp	.LBB230_110
.LBB230_110:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	movq	(%rax), %rax
	movb	(%rax), %cl
	movq	176(%rbp), %rax
	movq	%rax, %rdx
	incq	%rdx
	movq	%rdx, 176(%rbp)
	movb	%cl, (%rax)
	movq	328(%rbp), %rcx
.Ltmp358:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp359:
	jmp	.LBB230_111
.LBB230_111:
	jmp	.LBB230_112
.LBB230_112:
	cmpl	$0, 152(%rbp)
	jne	.LBB230_124
# %bb.113:
	jmp	.LBB230_114
.LBB230_114:                            # =>This Inner Loop Header: Depth=1
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp360:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp361:
	movb	%al, -57(%rbp)                  # 1-byte Spill
	jmp	.LBB230_115
.LBB230_115:                            #   in Loop: Header=BB230_114 Depth=1
	movb	-57(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -58(%rbp)                  # 1-byte Spill
	jne	.LBB230_116
	jmp	.LBB230_118
.LBB230_116:                            #   in Loop: Header=BB230_114 Depth=1
	movq	328(%rbp), %rcx
.Ltmp362:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp363:
	movb	%al, -59(%rbp)                  # 1-byte Spill
	jmp	.LBB230_117
.LBB230_117:                            #   in Loop: Header=BB230_114 Depth=1
	movb	-59(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	208(%rbp), %ecx
	cmpl	%ecx, %eax
	sete	%al
	movb	%al, -58(%rbp)                  # 1-byte Spill
.LBB230_118:                            #   in Loop: Header=BB230_114 Depth=1
	movb	-58(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_119
	jmp	.LBB230_121
.LBB230_119:                            #   in Loop: Header=BB230_114 Depth=1
	movl	148(%rbp), %eax
	decl	%eax
	movl	%eax, 148(%rbp)
	movb	$1, 159(%rbp)
	movq	328(%rbp), %rcx
.Ltmp404:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp405:
	jmp	.LBB230_120
.LBB230_120:                            #   in Loop: Header=BB230_114 Depth=1
	jmp	.LBB230_114
.LBB230_121:
	cmpl	$0, 148(%rbp)
	jge	.LBB230_123
# %bb.122:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 176(%rbp)
	movb	$48, (%rax)
	movl	148(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 148(%rbp)
.LBB230_123:
	jmp	.LBB230_124
.LBB230_124:
	jmp	.LBB230_125
.LBB230_125:                            # =>This Inner Loop Header: Depth=1
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp364:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp365:
	movb	%al, -60(%rbp)                  # 1-byte Spill
	jmp	.LBB230_126
.LBB230_126:                            #   in Loop: Header=BB230_125 Depth=1
	movb	-60(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -61(%rbp)                  # 1-byte Spill
	jne	.LBB230_127
	jmp	.LBB230_130
.LBB230_127:                            #   in Loop: Header=BB230_125 Depth=1
	movq	328(%rbp), %rcx
.Ltmp366:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp367:
	movb	%al, -62(%rbp)                  # 1-byte Spill
	jmp	.LBB230_128
.LBB230_128:                            #   in Loop: Header=BB230_125 Depth=1
.Ltmp368:
	movb	-62(%rbp), %dl                  # 1-byte Reload
	leaq	208(%rbp), %rcx
	callq	"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
.Ltmp369:
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	jmp	.LBB230_129
.LBB230_129:                            #   in Loop: Header=BB230_125 Depth=1
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 160(%rbp)
	cmpq	$22, %rax
	setb	%al
	movb	%al, -61(%rbp)                  # 1-byte Spill
.LBB230_130:                            #   in Loop: Header=BB230_125 Depth=1
	movb	-61(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_131
	jmp	.LBB230_135
.LBB230_131:                            #   in Loop: Header=BB230_125 Depth=1
	movl	152(%rbp), %eax
	cmpl	144(%rbp), %eax
	jge	.LBB230_133
# %bb.132:                              #   in Loop: Header=BB230_125 Depth=1
	movq	160(%rbp), %rcx
	leaq	"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	176(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 176(%rbp)
	movb	%cl, (%rax)
	movl	152(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 152(%rbp)
.LBB230_133:                            #   in Loop: Header=BB230_125 Depth=1
	movb	$1, 159(%rbp)
	movq	328(%rbp), %rcx
.Ltmp402:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp403:
	jmp	.LBB230_134
.LBB230_134:                            #   in Loop: Header=BB230_125 Depth=1
	jmp	.LBB230_125
.LBB230_135:
	testb	$1, 159(%rbp)
	je	.LBB230_177
# %bb.136:
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp370:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp371:
	movb	%al, -73(%rbp)                  # 1-byte Spill
	jmp	.LBB230_137
.LBB230_137:
	movb	-73(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_138
	jmp	.LBB230_177
.LBB230_138:
	movq	328(%rbp), %rcx
.Ltmp372:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp373:
	movb	%al, -74(%rbp)                  # 1-byte Spill
	jmp	.LBB230_139
.LBB230_139:
	movb	-74(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	235(%rbp), %ecx
	cmpl	%ecx, %eax
	je	.LBB230_142
# %bb.140:
	movq	328(%rbp), %rcx
.Ltmp374:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp375:
	movb	%al, -75(%rbp)                  # 1-byte Spill
	jmp	.LBB230_141
.LBB230_141:
	movb	-75(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	234(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_177
.LBB230_142:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$112, (%rax)
	movq	328(%rbp), %rcx
.Ltmp376:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp377:
	jmp	.LBB230_143
.LBB230_143:
	movb	$0, 159(%rbp)
	movl	$0, 152(%rbp)
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp378:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp379:
	movb	%al, -76(%rbp)                  # 1-byte Spill
	jmp	.LBB230_144
.LBB230_144:
	movb	-76(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_145
	jmp	.LBB230_155
.LBB230_145:
	movq	328(%rbp), %rcx
.Ltmp380:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp381:
	movb	%al, -77(%rbp)                  # 1-byte Spill
	jmp	.LBB230_146
.LBB230_146:
	movb	-77(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	231(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_149
# %bb.147:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$43, (%rax)
	movq	328(%rbp), %rcx
.Ltmp386:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp387:
	jmp	.LBB230_148
.LBB230_148:
	jmp	.LBB230_154
.LBB230_149:
	movq	328(%rbp), %rcx
.Ltmp382:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp383:
	movb	%al, -78(%rbp)                  # 1-byte Spill
	jmp	.LBB230_150
.LBB230_150:
	movb	-78(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	230(%rbp), %ecx
	cmpl	%ecx, %eax
	jne	.LBB230_153
# %bb.151:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	incq	%rcx
	movq	%rcx, 176(%rbp)
	movb	$45, (%rax)
	movq	328(%rbp), %rcx
.Ltmp384:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp385:
	jmp	.LBB230_152
.LBB230_152:
	jmp	.LBB230_153
.LBB230_153:
	jmp	.LBB230_154
.LBB230_154:
	jmp	.LBB230_155
.LBB230_155:
	jmp	.LBB230_156
.LBB230_156:                            # =>This Inner Loop Header: Depth=1
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp388:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp389:
	movb	%al, -79(%rbp)                  # 1-byte Spill
	jmp	.LBB230_157
.LBB230_157:                            #   in Loop: Header=BB230_156 Depth=1
	movb	-79(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -80(%rbp)                  # 1-byte Spill
	jne	.LBB230_158
	jmp	.LBB230_160
.LBB230_158:                            #   in Loop: Header=BB230_156 Depth=1
	movq	328(%rbp), %rcx
.Ltmp390:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp391:
	movb	%al, -81(%rbp)                  # 1-byte Spill
	jmp	.LBB230_159
.LBB230_159:                            #   in Loop: Header=BB230_156 Depth=1
	movb	-81(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movsbl	208(%rbp), %ecx
	cmpl	%ecx, %eax
	sete	%al
	movb	%al, -80(%rbp)                  # 1-byte Spill
.LBB230_160:                            #   in Loop: Header=BB230_156 Depth=1
	movb	-80(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_161
	jmp	.LBB230_163
.LBB230_161:                            #   in Loop: Header=BB230_156 Depth=1
	movb	$1, 159(%rbp)
	movq	328(%rbp), %rcx
.Ltmp400:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp401:
	jmp	.LBB230_162
.LBB230_162:                            #   in Loop: Header=BB230_156 Depth=1
	jmp	.LBB230_156
.LBB230_163:
	testb	$1, 159(%rbp)
	je	.LBB230_165
# %bb.164:
	movq	176(%rbp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 176(%rbp)
	movb	$48, (%rax)
.LBB230_165:
	jmp	.LBB230_166
.LBB230_166:                            # =>This Inner Loop Header: Depth=1
	movq	336(%rbp), %rdx
	movq	328(%rbp), %rcx
.Ltmp392:
	callq	"??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp393:
	movb	%al, -82(%rbp)                  # 1-byte Spill
	jmp	.LBB230_167
.LBB230_167:                            #   in Loop: Header=BB230_166 Depth=1
	movb	-82(%rbp), %cl                  # 1-byte Reload
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, -83(%rbp)                  # 1-byte Spill
	jne	.LBB230_168
	jmp	.LBB230_171
.LBB230_168:                            #   in Loop: Header=BB230_166 Depth=1
	movq	328(%rbp), %rcx
.Ltmp394:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp395:
	movb	%al, -84(%rbp)                  # 1-byte Spill
	jmp	.LBB230_169
.LBB230_169:                            #   in Loop: Header=BB230_166 Depth=1
.Ltmp396:
	movb	-84(%rbp), %dl                  # 1-byte Reload
	leaq	208(%rbp), %rcx
	callq	"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
.Ltmp397:
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	jmp	.LBB230_170
.LBB230_170:                            #   in Loop: Header=BB230_166 Depth=1
	movq	-96(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 160(%rbp)
	cmpq	$22, %rax
	setb	%al
	movb	%al, -83(%rbp)                  # 1-byte Spill
.LBB230_171:                            #   in Loop: Header=BB230_166 Depth=1
	movb	-83(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB230_172
	jmp	.LBB230_176
.LBB230_172:                            #   in Loop: Header=BB230_166 Depth=1
	cmpl	$8, 152(%rbp)
	jge	.LBB230_174
# %bb.173:                              #   in Loop: Header=BB230_166 Depth=1
	movq	160(%rbp), %rcx
	leaq	"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"(%rip), %rax
	movb	(%rax,%rcx), %cl
	movq	176(%rbp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 176(%rbp)
	movb	%cl, (%rax)
	movl	152(%rbp), %eax
	addl	$1, %eax
	movl	%eax, 152(%rbp)
.LBB230_174:                            #   in Loop: Header=BB230_166 Depth=1
	movb	$1, 159(%rbp)
	movq	328(%rbp), %rcx
.Ltmp398:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp399:
	jmp	.LBB230_175
.LBB230_175:                            #   in Loop: Header=BB230_166 Depth=1
	jmp	.LBB230_166
.LBB230_176:
	jmp	.LBB230_177
.LBB230_177:
	testb	$1, 175(%rbp)
	jne	.LBB230_179
# %bb.178:
	testb	$1, 159(%rbp)
	jne	.LBB230_180
.LBB230_179:
	movq	320(%rbp), %rax
	movq	%rax, 176(%rbp)
.LBB230_180:
	movq	176(%rbp), %rax
	movb	$0, (%rax)
	movl	148(%rbp), %ecx
	movq	408(%rbp), %rax
	movl	%ecx, (%rax)
	leaq	256(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	xorl	%eax, %eax
	addq	$480, %rsp                      # imm = 0x1E0
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$9@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$9@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$9@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB230_9:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	288(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$10@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$10@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB230_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	184(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$100@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$100@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$100@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB230_100:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.def	"?dtor$181@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$181@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA":
.seh_proc "?dtor$181@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"
.LBB230_181:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	256(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end40:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.p2align	2
"$cppxdata$?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z":
	.long	429065506                       # MagicNumber
	.long	4                               # MaxState
	.long	("$stateUnwindMap$?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	8                               # IPMapEntries
	.long	("$ip2state$?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z")@IMGREL # IPToStateXData
	.long	472                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z":
	.long	-1                              # ToState
	.long	"?dtor$9@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$181@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$100@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$10@?0??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z@4HA"@IMGREL # Action
"$ip2state$?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z":
	.long	.Lfunc_begin40@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp298@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp299@IMGREL+1               # IP
	.long	-1                              # ToState
	.long	.Ltmp300@IMGREL+1               # IP
	.long	3                               # ToState
	.long	.Ltmp302@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp332@IMGREL+1               # IP
	.long	2                               # ToState
	.long	.Ltmp350@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp399@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
                                        # -- End function
	.def	"??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
	.globl	"??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z" # -- Begin function ??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z
	.p2align	4, 0x90
"??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z": # @"??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
.seh_proc "??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	addq	$15, %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
	.globl	"??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z" # -- Begin function ??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z
	.p2align	4, 0x90
"??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z": # @"??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
.seh_proc "??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
	.globl	"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z" # -- Begin function ??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z
	.p2align	4, 0x90
"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z": # @"??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
.seh_proc "??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rdx
	addq	$14, %rdx
	movq	40(%rsp), %rcx
	leaq	55(%rsp), %r8
	callq	"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
	movq	40(%rsp), %rcx
	subq	%rcx, %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?decimal_point@?$numpunct@D@std@@QEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?decimal_point@?$numpunct@D@std@@QEBADXZ"
	.globl	"?decimal_point@?$numpunct@D@std@@QEBADXZ" # -- Begin function ?decimal_point@?$numpunct@D@std@@QEBADXZ
	.p2align	4, 0x90
"?decimal_point@?$numpunct@D@std@@QEBADXZ": # @"?decimal_point@?$numpunct@D@std@@QEBADXZ"
.seh_proc "?decimal_point@?$numpunct@D@std@@QEBADXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*24(%rax)
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
	.globl	"??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z" # -- Begin function ??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z
	.p2align	4, 0x90
"??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z": # @"??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
.seh_proc "??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	addq	$29, %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
	.globl	"??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z" # -- Begin function ??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z
	.p2align	4, 0x90
"??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z": # @"??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
.seh_proc "??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
	.globl	"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z" # -- Begin function ??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z
	.p2align	4, 0x90
"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z": # @"??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
.seh_proc "??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rdx
	addq	$28, %rdx
	movq	40(%rsp), %rcx
	leaq	55(%rsp), %r8
	callq	"??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
	movq	40(%rsp), %rcx
	subq	%rcx, %rax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	__real@41200000                 # -- Begin function ?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z
	.section	.rdata,"dr",discard,__real@41200000
	.p2align	2
__real@41200000:
	.long	0x41200000                      # float 10
	.section	.text,"xr",discard,"?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z"
	.globl	"?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z"
	.p2align	4, 0x90
"?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z": # @"?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z"
.seh_proc "?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r9, 80(%rsp)
	movl	%r8d, 76(%rsp)
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	callq	_errno
	movq	%rax, 48(%rsp)
	movq	48(%rsp), %rax
	movl	(%rax), %eax
	movl	%eax, 44(%rsp)
	movq	48(%rsp), %rax
	movl	$0, (%rax)
	movq	64(%rsp), %rdx
	movq	56(%rsp), %rcx
	callq	strtof
	movss	%xmm0, 40(%rsp)
	movq	48(%rsp), %rax
	movl	(%rax), %ecx
	movq	80(%rsp), %rax
	movl	%ecx, (%rax)
	movl	44(%rsp), %ecx
	movq	48(%rsp), %rax
	movl	%ecx, (%rax)
	cmpl	$0, 76(%rsp)
	je	.LBB238_2
# %bb.1:
	cvtsi2ssl	76(%rsp), %xmm1
	movss	__real@41200000(%rip), %xmm0    # xmm0 = mem[0],zero,zero,zero
	callq	powf
	mulss	40(%rsp), %xmm0
	movss	%xmm0, 40(%rsp)
.LBB238_2:
	movss	40(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	ldexpf;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,ldexpf
	.globl	ldexpf                          # -- Begin function ldexpf
	.p2align	4, 0x90
ldexpf:                                 # @ldexpf
.seh_proc ldexpf
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movl	%edx, 36(%rsp)
	movss	%xmm0, 32(%rsp)
	movl	36(%rsp), %edx
	movss	32(%rsp), %xmm0                 # xmm0 = mem[0],zero,zero,zero
	cvtss2sd	%xmm0, %xmm0
	callq	ldexp
	cvtsd2ss	%xmm0, %xmm0
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*48(%rax)
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z"
	.globl	"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z" # -- Begin function ??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z
	.p2align	4, 0x90
"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z": # @"??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z"
.seh_proc "??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	48(%rsp), %rdx
	callq	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	(%rcx), %rax
	callq	*56(%rax)
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.globl	"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z" # -- Begin function ??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z
	.p2align	4, 0x90
"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z": # @"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
.Lfunc_begin41:
.seh_proc "??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$272, %rsp                      # imm = 0x110
	.seh_stackalloc 272
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 136(%rbp)
	movb	192(%rbp), %al
	andb	$1, %al
	movb	%al, 135(%rbp)
	movq	%r9, 120(%rbp)
	movq	%r8, 112(%rbp)
	movq	%rdx, 104(%rbp)
	movq	%rcx, 96(%rbp)
	movq	$0, 88(%rbp)
.LBB243_1:                              # =>This Inner Loop Header: Depth=1
	movq	120(%rbp), %rax
	movq	88(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	cmpl	$0, %eax
	je	.LBB243_5
# %bb.2:                                #   in Loop: Header=BB243_1 Depth=1
	movq	120(%rbp), %rax
	movq	88(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	movq	120(%rbp), %rcx
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	jne	.LBB243_4
# %bb.3:                                #   in Loop: Header=BB243_1 Depth=1
	movq	112(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 112(%rbp)
.LBB243_4:                              #   in Loop: Header=BB243_1 Depth=1
	movq	88(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 88(%rbp)
	jmp	.LBB243_1
.LBB243_5:
	movq	112(%rbp), %rdx
	xorl	%eax, %eax
	movb	%al, %r8b
	leaq	56(%rbp), %rcx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	leaq	32(%rbp), %rcx
	movq	%rcx, -24(%rbp)                 # 8-byte Spill
	callq	"??0locale@std@@QEAA@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
.Ltmp408:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp409:
	movq	%rax, -16(%rbp)                 # 8-byte Spill
	jmp	.LBB243_6
.LBB243_6:
	leaq	32(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-16(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 48(%rbp)
	movl	$-2, 28(%rbp)
	movq	$1, 16(%rbp)
.LBB243_7:                              # =>This Loop Header: Depth=1
                                        #     Child Loop BB243_8 Depth 2
                                        #       Child Loop BB243_10 Depth 3
	movb	$0, 15(%rbp)
	movq	$0, (%rbp)
	movq	$0, -8(%rbp)
.LBB243_8:                              #   Parent Loop BB243_7 Depth=1
                                        # =>  This Loop Header: Depth=2
                                        #       Child Loop BB243_10 Depth 3
	movq	-8(%rbp), %rax
	cmpq	112(%rbp), %rax
	jae	.LBB243_40
# %bb.9:                                #   in Loop: Header=BB243_8 Depth=2
	jmp	.LBB243_10
.LBB243_10:                             #   Parent Loop BB243_7 Depth=1
                                        #     Parent Loop BB243_8 Depth=2
                                        # =>    This Inner Loop Header: Depth=3
	movq	120(%rbp), %rax
	movq	(%rbp), %rcx
	movsbl	(%rax,%rcx), %ecx
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	cmpl	$0, %ecx
	movb	%al, -25(%rbp)                  # 1-byte Spill
	je	.LBB243_12
# %bb.11:                               #   in Loop: Header=BB243_10 Depth=3
	movq	120(%rbp), %rax
	movq	(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	movq	120(%rbp), %rcx
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	setne	%al
	movb	%al, -25(%rbp)                  # 1-byte Spill
.LBB243_12:                             #   in Loop: Header=BB243_10 Depth=3
	movb	-25(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB243_13
	jmp	.LBB243_15
.LBB243_13:                             #   in Loop: Header=BB243_10 Depth=3
	movq	(%rbp), %rax
	addq	$1, %rax
	movq	%rax, (%rbp)
	jmp	.LBB243_10
.LBB243_15:                             #   in Loop: Header=BB243_8 Depth=2
	movq	-8(%rbp), %rdx
	leaq	56(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbl	(%rax), %eax
	cmpl	$0, %eax
	je	.LBB243_17
# %bb.16:                               #   in Loop: Header=BB243_8 Depth=2
	movq	-8(%rbp), %rdx
	leaq	56(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsbq	(%rax), %rax
	addq	(%rbp), %rax
	movq	%rax, (%rbp)
	jmp	.LBB243_39
.LBB243_17:                             #   in Loop: Header=BB243_8 Depth=2
	movq	120(%rbp), %rax
	movq	16(%rbp), %rcx
	addq	(%rbp), %rcx
	movq	%rcx, (%rbp)
	movsbl	(%rax,%rcx), %eax
	movq	120(%rbp), %rcx
	movsbl	(%rcx), %ecx
	cmpl	%ecx, %eax
	je	.LBB243_19
# %bb.18:                               #   in Loop: Header=BB243_8 Depth=2
	movq	120(%rbp), %rax
	movq	(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	cmpl	$0, %eax
	jne	.LBB243_23
.LBB243_19:                             #   in Loop: Header=BB243_8 Depth=2
	cmpq	$127, 16(%rbp)
	jae	.LBB243_21
# %bb.20:                               #   in Loop: Header=BB243_8 Depth=2
	movq	16(%rbp), %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	jmp	.LBB243_22
.LBB243_21:                             #   in Loop: Header=BB243_8 Depth=2
	movl	$127, %eax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	jmp	.LBB243_22
.LBB243_22:                             #   in Loop: Header=BB243_8 Depth=2
	movq	-40(%rbp), %rax                 # 8-byte Reload
                                        # kill: def $al killed $al killed $rax
	movb	%al, -41(%rbp)                  # 1-byte Spill
	movq	-8(%rbp), %rdx
	leaq	56(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movb	-41(%rbp), %cl                  # 1-byte Reload
	movb	%cl, (%rax)
	movq	-8(%rbp), %rax
                                        # kill: def $eax killed $eax killed $rax
	movl	%eax, 28(%rbp)
	jmp	.LBB243_38
.LBB243_23:                             #   in Loop: Header=BB243_8 Depth=2
	movq	104(%rbp), %rdx
	movq	96(%rbp), %rcx
.Ltmp414:
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp415:
	movb	%al, -42(%rbp)                  # 1-byte Spill
	jmp	.LBB243_24
.LBB243_24:                             #   in Loop: Header=BB243_8 Depth=2
	movb	-42(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB243_32
	jmp	.LBB243_25
.LBB243_25:                             #   in Loop: Header=BB243_8 Depth=2
	testb	$1, 135(%rbp)
	je	.LBB243_28
# %bb.26:                               #   in Loop: Header=BB243_8 Depth=2
	movq	120(%rbp), %rax
	movq	(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	movl	%eax, -48(%rbp)                 # 4-byte Spill
	movq	96(%rbp), %rcx
.Ltmp422:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp423:
	movb	%al, -43(%rbp)                  # 1-byte Spill
	jmp	.LBB243_27
.LBB243_27:                             #   in Loop: Header=BB243_8 Depth=2
	movl	-48(%rbp), %eax                 # 4-byte Reload
	movb	-43(%rbp), %cl                  # 1-byte Reload
	movsbl	%cl, %ecx
	cmpl	%ecx, %eax
	jne	.LBB243_32
	jmp	.LBB243_36
.LBB243_28:                             #   in Loop: Header=BB243_8 Depth=2
	movq	48(%rbp), %rcx
	movq	120(%rbp), %rax
	movq	(%rbp), %rdx
	movb	(%rax,%rdx), %dl
.Ltmp416:
	callq	"?tolower@?$ctype@D@std@@QEBADD@Z"
.Ltmp417:
	movb	%al, -49(%rbp)                  # 1-byte Spill
	jmp	.LBB243_29
.LBB243_29:                             #   in Loop: Header=BB243_8 Depth=2
	movb	-49(%rbp), %al                  # 1-byte Reload
	movsbl	%al, %eax
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	movq	48(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	96(%rbp), %rcx
.Ltmp418:
	callq	"??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
.Ltmp419:
	movb	%al, -50(%rbp)                  # 1-byte Spill
	jmp	.LBB243_30
.LBB243_30:                             #   in Loop: Header=BB243_8 Depth=2
.Ltmp420:
	movb	-50(%rbp), %dl                  # 1-byte Reload
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	callq	"?tolower@?$ctype@D@std@@QEBADD@Z"
.Ltmp421:
	movb	%al, -69(%rbp)                  # 1-byte Spill
	jmp	.LBB243_31
.LBB243_31:                             #   in Loop: Header=BB243_8 Depth=2
	movl	-68(%rbp), %eax                 # 4-byte Reload
	movb	-69(%rbp), %cl                  # 1-byte Reload
	movsbl	%cl, %ecx
	cmpl	%ecx, %eax
	je	.LBB243_36
.LBB243_32:                             #   in Loop: Header=BB243_8 Depth=2
	cmpq	$127, 16(%rbp)
	jae	.LBB243_34
# %bb.33:                               #   in Loop: Header=BB243_8 Depth=2
	movq	16(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB243_35
.LBB243_34:                             #   in Loop: Header=BB243_8 Depth=2
	movl	$127, %eax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB243_35
.LBB243_35:                             #   in Loop: Header=BB243_8 Depth=2
	movq	-80(%rbp), %rax                 # 8-byte Reload
                                        # kill: def $al killed $al killed $rax
	movb	%al, -81(%rbp)                  # 1-byte Spill
	movq	-8(%rbp), %rdx
	leaq	56(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movb	-81(%rbp), %cl                  # 1-byte Reload
	movb	%cl, (%rax)
	jmp	.LBB243_37
.LBB243_36:                             #   in Loop: Header=BB243_8 Depth=2
	movb	$1, 15(%rbp)
.LBB243_37:                             #   in Loop: Header=BB243_8 Depth=2
	jmp	.LBB243_38
.LBB243_38:                             #   in Loop: Header=BB243_8 Depth=2
	jmp	.LBB243_39
.LBB243_39:                             #   in Loop: Header=BB243_8 Depth=2
	movq	-8(%rbp), %rax
	addq	$1, %rax
	movq	%rax, -8(%rbp)
	jmp	.LBB243_8
.LBB243_40:                             #   in Loop: Header=BB243_7 Depth=1
	testb	$1, 15(%rbp)
	je	.LBB243_43
# %bb.41:                               #   in Loop: Header=BB243_7 Depth=1
	movq	104(%rbp), %rdx
	movq	96(%rbp), %rcx
.Ltmp410:
	callq	"??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
.Ltmp411:
	movb	%al, -82(%rbp)                  # 1-byte Spill
	jmp	.LBB243_42
.LBB243_42:                             #   in Loop: Header=BB243_7 Depth=1
	movb	-82(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB243_43
	jmp	.LBB243_44
.LBB243_43:
	movl	28(%rbp), %eax
	movl	%eax, -88(%rbp)                 # 4-byte Spill
	leaq	56(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movl	-88(%rbp), %eax                 # 4-byte Reload
	addq	$272, %rsp                      # imm = 0x110
	popq	%rbp
	retq
.LBB243_44:                             #   in Loop: Header=BB243_7 Depth=1
	movq	16(%rbp), %rax
	incq	%rax
	movq	%rax, 16(%rbp)
	movq	96(%rbp), %rcx
.Ltmp412:
	callq	"??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.Ltmp413:
	jmp	.LBB243_45
.LBB243_45:                             #   in Loop: Header=BB243_7 Depth=1
	movl	$-1, 28(%rbp)
	jmp	.LBB243_7
	.seh_handlerdata
	.long	("$cppxdata$??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z")@IMGREL
	.section	.text,"xr",discard,"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.seh_endproc
	.def	"?dtor$14@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$14@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA":
.seh_proc "?dtor$14@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA"
.LBB243_14:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	32(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.seh_endproc
	.def	"?dtor$46@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$46@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA":
.seh_proc "?dtor$46@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA"
.LBB243_46:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	56(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end41:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.p2align	2
"$cppxdata$??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z")@IMGREL # IPToStateXData
	.long	264                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z":
	.long	-1                              # ToState
	.long	"?dtor$46@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$14@?0???$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z@4HA"@IMGREL # Action
"$ip2state$??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z":
	.long	.Lfunc_begin41@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp408@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp414@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp413@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
                                        # -- End function
	.def	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	.globl	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ" # -- Begin function ?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ
	.p2align	4, 0x90
"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ": # @"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
.seh_proc "?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"
	.globl	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z" # -- Begin function ?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z
	.p2align	4, 0x90
"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z": # @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"
.seh_proc "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rax
	movq	16(%rax), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rcx
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	48(%rsp), %r8                   # 8-byte Reload
	movq	%rax, %rdx
	callq	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"
	.globl	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z" # -- Begin function ?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z
	.p2align	4, 0x90
"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z": # @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"
.seh_proc "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"
# %bb.0:
	subq	$104, %rsp
	.seh_stackalloc 104
	.seh_endprologue
	movq	%r8, 88(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	72(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	16(%rcx), %rax
	movq	%rax, 64(%rsp)
	movq	88(%rsp), %rax
	movq	24(%rcx), %rcx
	subq	64(%rsp), %rcx
	cmpq	%rcx, %rax
	ja	.LBB246_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	64(%rsp), %rax
	addq	88(%rsp), %rax
	movq	%rax, 16(%rcx)
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	movq	%rax, 56(%rsp)
	movq	88(%rsp), %r8
	movq	80(%rsp), %rdx
	movq	56(%rsp), %rcx
	addq	64(%rsp), %rcx
	callq	"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	$0, 55(%rsp)
	movq	56(%rsp), %rcx
	movq	64(%rsp), %rax
	addq	88(%rsp), %rax
	addq	%rax, %rcx
	leaq	55(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 96(%rsp)
	jmp	.LBB246_3
.LBB246_2:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rax
	movq	80(%rsp), %r9
	movq	88(%rsp), %rdx
	movb	48(%rsp), %r8b
	movq	%rax, 32(%rsp)
	callq	"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"
	movq	%rax, 96(%rsp)
.LBB246_3:
	movq	96(%rsp), %rax
	addq	$104, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"
	.globl	"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z" # -- Begin function ??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z
	.p2align	4, 0x90
"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z": # @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"
.seh_proc "??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"
# %bb.0:
	subq	$184, %rsp
	.seh_stackalloc 184
	.seh_endprologue
	movq	224(%rsp), %rax
	movb	%r8b, 176(%rsp)
	movq	%r9, 168(%rsp)
	movq	%rdx, 160(%rsp)
	movq	%rcx, 152(%rsp)
	movq	152(%rsp), %rcx
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	movq	%rcx, 144(%rsp)
	movq	144(%rsp), %rax
	movq	16(%rax), %rax
	movq	%rax, 136(%rsp)
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	subq	136(%rsp), %rax
	cmpq	160(%rsp), %rax
	jae	.LBB247_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB247_2:
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	136(%rsp), %rax
	addq	160(%rsp), %rax
	movq	%rax, 128(%rsp)
	movq	144(%rsp), %rax
	movq	24(%rax), %rax
	movq	%rax, 120(%rsp)
	movq	128(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 112(%rsp)
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 104(%rsp)
	movq	104(%rsp), %rcx
	movq	112(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 96(%rsp)
	movq	144(%rsp), %rcx
	callq	"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	movq	128(%rsp), %rcx
	movq	144(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	112(%rsp), %rcx
	movq	144(%rsp), %rax
	movq	%rcx, 24(%rax)
	movq	96(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, 88(%rsp)
	movl	$16, %eax
	cmpq	120(%rsp), %rax
	ja	.LBB247_4
# %bb.3:
	movq	144(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 80(%rsp)
	movq	224(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	168(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	136(%rsp), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	80(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	48(%rsp), %r9                   # 8-byte Reload
	movq	56(%rsp), %r10                  # 8-byte Reload
	movq	%rax, %r8
	movq	64(%rsp), %rax                  # 8-byte Reload
	movq	88(%rsp), %rdx
	leaq	176(%rsp), %rcx
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"
	movq	104(%rsp), %rcx
	movq	120(%rsp), %r8
	addq	$1, %r8
	movq	80(%rsp), %rdx
	callq	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
	movq	96(%rsp), %rcx
	movq	144(%rsp), %rax
	movq	%rcx, (%rax)
	jmp	.LBB247_5
.LBB247_4:
	movq	224(%rsp), %rax
	movq	168(%rsp), %r10
	movq	136(%rsp), %r9
	movq	144(%rsp), %r8
	movq	88(%rsp), %rdx
	leaq	176(%rsp), %rcx
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"
	movq	144(%rsp), %rcx
	leaq	96(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
.LBB247_5:
	movq	72(%rsp), %rax                  # 8-byte Reload
	addq	$184, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"
	.globl	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z" # -- Begin function ??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z
	.p2align	4, 0x90
"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z": # @"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"
.seh_proc "??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	120(%rsp), %rax
	movq	112(%rsp), %rax
	movq	%r9, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	64(%rsp), %r8
	movq	56(%rsp), %rdx
	movq	48(%rsp), %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movq	120(%rsp), %r8
	movq	112(%rsp), %rdx
	movq	48(%rsp), %rcx
	addq	64(%rsp), %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	$0, 39(%rsp)
	movq	48(%rsp), %rcx
	movq	64(%rsp), %rax
	addq	120(%rsp), %rax
	addq	%rax, %rcx
	leaq	39(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0locale@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0locale@std@@QEAA@XZ"
	.globl	"??0locale@std@@QEAA@XZ"        # -- Begin function ??0locale@std@@QEAA@XZ
	.p2align	4, 0x90
"??0locale@std@@QEAA@XZ":               # @"??0locale@std@@QEAA@XZ"
.Lfunc_begin42:
.seh_proc "??0locale@std@@QEAA@XZ"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	addq	$8, %rax
	movq	%rax, -32(%rbp)                 # 8-byte Spill
.Ltmp424:
	movb	$1, %cl
	callq	"?_Init@locale@std@@CAPEAV_Locimp@12@_N@Z"
.Ltmp425:
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	jmp	.LBB249_1
.LBB249_1:
	movq	-40(%rbp), %rax                 # 8-byte Reload
	movq	-32(%rbp), %rcx                 # 8-byte Reload
	movq	-24(%rbp), %rdx                 # 8-byte Reload
	movq	%rdx, (%rcx)
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0locale@std@@QEAA@XZ")@IMGREL
	.section	.text,"xr",discard,"??0locale@std@@QEAA@XZ"
	.seh_endproc
	.def	"?dtor$2@?0???0locale@std@@QEAA@XZ@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0locale@std@@QEAA@XZ@4HA":
.seh_proc "?dtor$2@?0???0locale@std@@QEAA@XZ@4HA"
.LBB249_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end42:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0locale@std@@QEAA@XZ"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0locale@std@@QEAA@XZ"
	.p2align	2
"$cppxdata$??0locale@std@@QEAA@XZ":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0locale@std@@QEAA@XZ")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0locale@std@@QEAA@XZ")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0locale@std@@QEAA@XZ":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0locale@std@@QEAA@XZ@4HA"@IMGREL # Action
"$ip2state$??0locale@std@@QEAA@XZ":
	.long	.Lfunc_begin42@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp424@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp425@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0locale@std@@QEAA@XZ"
                                        # -- End function
	.def	"?tolower@?$ctype@D@std@@QEBADD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?tolower@?$ctype@D@std@@QEBADD@Z"
	.globl	"?tolower@?$ctype@D@std@@QEBADD@Z" # -- Begin function ?tolower@?$ctype@D@std@@QEBADD@Z
	.p2align	4, 0x90
"?tolower@?$ctype@D@std@@QEBADD@Z":     # @"?tolower@?$ctype@D@std@@QEBADD@Z"
.seh_proc "?tolower@?$ctype@D@std@@QEBADD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movb	55(%rsp), %dl
	movq	(%rcx), %rax
	callq	*32(%rax)
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0locale@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0locale@std@@QEAA@AEBV01@@Z"
	.globl	"??0locale@std@@QEAA@AEBV01@@Z" # -- Begin function ??0locale@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0locale@std@@QEAA@AEBV01@@Z":        # @"??0locale@std@@QEAA@AEBV01@@Z"
.seh_proc "??0locale@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rcx
	movq	8(%rcx), %rcx
	movq	%rcx, 8(%rax)
	movq	8(%rax), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	.globl	"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z" # -- Begin function ?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z
	.p2align	4, 0x90
"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z": # @"?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.seh_proc "?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	andb	$1, %r8b
	movb	%r8b, 55(%rsp)
	movl	%edx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movb	55(%rsp), %r8b
	movl	48(%rsp), %edx
	movq	72(%rcx), %r10
	movl	$4, %eax
	xorl	%r9d, %r9d
	cmpq	$0, %r10
	cmovnel	%r9d, %eax
	orl	%eax, %edx
	andb	$1, %r8b
	callq	"?clear@ios_base@std@@QEAAXH_N@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?clear@ios_base@std@@QEAAXH_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?clear@ios_base@std@@QEAAXH_N@Z"
	.globl	"?clear@ios_base@std@@QEAAXH_N@Z" # -- Begin function ?clear@ios_base@std@@QEAAXH_N@Z
	.p2align	4, 0x90
"?clear@ios_base@std@@QEAAXH_N@Z":      # @"?clear@ios_base@std@@QEAAXH_N@Z"
.seh_proc "?clear@ios_base@std@@QEAAXH_N@Z"
# %bb.0:
	subq	$120, %rsp
	.seh_stackalloc 120
	.seh_endprologue
	andb	$1, %r8b
	movb	%r8b, 119(%rsp)
	movl	%edx, 112(%rsp)
	movq	%rcx, 104(%rsp)
	movq	104(%rsp), %rcx
	movl	112(%rsp), %eax
	andl	$23, %eax
	movl	%eax, 112(%rsp)
	movl	112(%rsp), %eax
	movl	%eax, 16(%rcx)
	movl	112(%rsp), %eax
	andl	20(%rcx), %eax
	movl	%eax, 100(%rsp)
	cmpl	$0, 100(%rsp)
	je	.LBB253_10
# %bb.1:
	testb	$1, 119(%rsp)
	je	.LBB253_3
# %bb.2:
	xorl	%eax, %eax
	movl	%eax, %edx
	movq	%rdx, %rcx
	callq	_CxxThrowException
.LBB253_3:
	movl	100(%rsp), %eax
	andl	$4, %eax
	cmpl	$0, %eax
	je	.LBB253_5
# %bb.4:
	leaq	"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"(%rip), %rax
	movq	%rax, 88(%rsp)
	jmp	.LBB253_9
.LBB253_5:
	movl	100(%rsp), %eax
	andl	$2, %eax
	cmpl	$0, %eax
	je	.LBB253_7
# %bb.6:
	leaq	"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"(%rip), %rax
	movq	%rax, 88(%rsp)
	jmp	.LBB253_8
.LBB253_7:
	leaq	"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@"(%rip), %rax
	movq	%rax, 88(%rsp)
.LBB253_8:
	jmp	.LBB253_9
.LBB253_9:
	leaq	32(%rsp), %rcx
	movl	$1, %edx
	callq	"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"
	movq	88(%rsp), %rdx
	leaq	48(%rsp), %rcx
	leaq	32(%rsp), %r8
	callq	"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"
	leaq	48(%rsp), %rcx
	leaq	"_TI5?AVfailure@ios_base@std@@"(%rip), %rdx
	callq	_CxxThrowException
.LBB253_10:
	nop
	addq	$120, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"
	.globl	"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z" # -- Begin function ?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z
	.p2align	4, 0x90
"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z": # @"?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"
.seh_proc "?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	%edx, 60(%rsp)
	callq	"?iostream_category@std@@YAAEBVerror_category@1@XZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %r8
	movl	60(%rsp), %edx
	callq	"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z"
                                        # kill: def $rcx killed $rax
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"
	.globl	"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z" # -- Begin function ??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z
	.p2align	4, 0x90
"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z": # @"??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"
.seh_proc "??0failure@ios_base@std@@QEAA@PEBDAEBVerror_code@2@@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	72(%rsp), %r8
	movq	80(%rsp), %rax
	movq	(%rax), %rdx
	movq	%rdx, 48(%rsp)
	movq	8(%rax), %rax
	movq	%rax, 56(%rsp)
	leaq	48(%rsp), %rdx
	callq	"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7failure@ios_base@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0failure@ios_base@std@@QEAA@AEBV012@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0failure@ios_base@std@@QEAA@AEBV012@@Z"
	.globl	"??0failure@ios_base@std@@QEAA@AEBV012@@Z" # -- Begin function ??0failure@ios_base@std@@QEAA@AEBV012@@Z
	.p2align	4, 0x90
"??0failure@ios_base@std@@QEAA@AEBV012@@Z": # @"??0failure@ios_base@std@@QEAA@AEBV012@@Z"
.seh_proc "??0failure@ios_base@std@@QEAA@AEBV012@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0system_error@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7failure@ios_base@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0system_error@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0system_error@std@@QEAA@AEBV01@@Z"
	.globl	"??0system_error@std@@QEAA@AEBV01@@Z" # -- Begin function ??0system_error@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0system_error@std@@QEAA@AEBV01@@Z":  # @"??0system_error@std@@QEAA@AEBV01@@Z"
.seh_proc "??0system_error@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0_System_error@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7system_error@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_System_error@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_System_error@std@@QEAA@AEBV01@@Z"
	.globl	"??0_System_error@std@@QEAA@AEBV01@@Z" # -- Begin function ??0_System_error@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0_System_error@std@@QEAA@AEBV01@@Z": # @"??0_System_error@std@@QEAA@AEBV01@@Z"
.seh_proc "??0_System_error@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0runtime_error@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7_System_error@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	movq	48(%rsp), %rcx
	movq	24(%rcx), %rdx
	movq	%rdx, 24(%rax)
	movq	32(%rcx), %rcx
	movq	%rcx, 32(%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0runtime_error@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0runtime_error@std@@QEAA@AEBV01@@Z"
	.globl	"??0runtime_error@std@@QEAA@AEBV01@@Z" # -- Begin function ??0runtime_error@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0runtime_error@std@@QEAA@AEBV01@@Z": # @"??0runtime_error@std@@QEAA@AEBV01@@Z"
.seh_proc "??0runtime_error@std@@QEAA@AEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??0exception@std@@QEAA@AEBV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7runtime_error@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1failure@ios_base@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1failure@ios_base@std@@UEAA@XZ"
	.globl	"??1failure@ios_base@std@@UEAA@XZ" # -- Begin function ??1failure@ios_base@std@@UEAA@XZ
	.p2align	4, 0x90
"??1failure@ios_base@std@@UEAA@XZ":     # @"??1failure@ios_base@std@@UEAA@XZ"
.seh_proc "??1failure@ios_base@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1system_error@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?iostream_category@std@@YAAEBVerror_category@1@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?iostream_category@std@@YAAEBVerror_category@1@XZ"
	.globl	"?iostream_category@std@@YAAEBVerror_category@1@XZ" # -- Begin function ?iostream_category@std@@YAAEBVerror_category@1@XZ
	.p2align	4, 0x90
"?iostream_category@std@@YAAEBVerror_category@1@XZ": # @"?iostream_category@std@@YAAEBVerror_category@1@XZ"
.seh_proc "?iostream_category@std@@YAAEBVerror_category@1@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	callq	"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z"
	.globl	"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z" # -- Begin function ??0error_code@std@@QEAA@HAEBVerror_category@1@@Z
	.p2align	4, 0x90
"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z": # @"??0error_code@std@@QEAA@HAEBVerror_category@1@@Z"
.seh_proc "??0error_code@std@@QEAA@HAEBVerror_category@1@@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%r8, 16(%rsp)
	movl	%edx, 12(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	12(%rsp), %ecx
	movl	%ecx, (%rax)
	movq	16(%rsp), %rcx
	movq	%rcx, 8(%rax)
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"
	.globl	"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ" # -- Begin function ??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ
	.p2align	4, 0x90
"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ": # @"??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"
.seh_proc "??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movl	"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"(%rip), %eax
	movl	_tls_index(%rip), %ecx
	movl	%ecx, %edx
	movq	%gs:88, %rcx
	movq	(%rcx,%rdx,8), %rcx
	movl	_Init_thread_epoch@SECREL32(%rcx), %ecx
	cmpl	%ecx, %eax
	jle	.LBB263_3
# %bb.1:
	leaq	"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"(%rip), %rcx
	callq	_Init_thread_header
	cmpl	$-1, "?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"(%rip)
	jne	.LBB263_3
# %bb.2:
	leaq	"??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ"(%rip), %rcx
	callq	atexit
	leaq	"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"(%rip), %rcx
	callq	_Init_thread_footer
.LBB263_3:
	leaq	"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A"(%rip), %rax
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1_Iostream_error_category2@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_Iostream_error_category2@std@@UEAA@XZ"
	.globl	"??1_Iostream_error_category2@std@@UEAA@XZ" # -- Begin function ??1_Iostream_error_category2@std@@UEAA@XZ
	.p2align	4, 0x90
"??1_Iostream_error_category2@std@@UEAA@XZ": # @"??1_Iostream_error_category2@std@@UEAA@XZ"
.seh_proc "??1_Iostream_error_category2@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1error_category@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ";
	.scl	3;
	.type	32;
	.endef
	.text
	.p2align	4, 0x90                         # -- Begin function ??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ
"??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ": # @"??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ"
.seh_proc "??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	leaq	"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A"(%rip), %rcx
	callq	"??1_Iostream_error_category2@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z"
	.globl	"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z" # -- Begin function ??_G_Iostream_error_category2@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z": # @"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z"
.seh_proc "??_G_Iostream_error_category2@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1_Iostream_error_category2@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB266_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB266_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?name@_Iostream_error_category2@std@@UEBAPEBDXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?name@_Iostream_error_category2@std@@UEBAPEBDXZ"
	.globl	"?name@_Iostream_error_category2@std@@UEBAPEBDXZ" # -- Begin function ?name@_Iostream_error_category2@std@@UEBAPEBDXZ
	.p2align	4, 0x90
"?name@_Iostream_error_category2@std@@UEBAPEBDXZ": # @"?name@_Iostream_error_category2@std@@UEBAPEBDXZ"
.seh_proc "?name@_Iostream_error_category2@std@@UEBAPEBDXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	leaq	"??_C@_08LLGCOLLL@iostream?$AA@"(%rip), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z"
	.globl	"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z" # -- Begin function ?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z
	.p2align	4, 0x90
"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z": # @"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z"
.seh_proc "?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	%rdx, 80(%rsp)
	movl	%r8d, 76(%rsp)
	movq	%rcx, 64(%rsp)
	cmpl	$1, 76(%rsp)
	jne	.LBB268_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	$21, 56(%rsp)
	leaq	"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB"(%rip), %rdx
	movl	$21, %r8d
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
	jmp	.LBB268_3
.LBB268_2:
	movl	76(%rsp), %ecx
	callq	"?_Syserror_map@std@@YAPEBDH@Z"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %rdx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
.LBB268_3:
	movq	48(%rsp), %rax                  # 8-byte Reload
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z"
	.globl	"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z" # -- Begin function ?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z
	.p2align	4, 0x90
"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z": # @"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z"
.seh_proc "?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, %rax
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	%rcx, %rdx
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	%r8d, 60(%rsp)
	movq	%rax, 48(%rsp)
	movq	48(%rsp), %r8
	movl	60(%rsp), %edx
	callq	"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z"
	.globl	"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z" # -- Begin function ?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z
	.p2align	4, 0x90
"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z": # @"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z"
.seh_proc "?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%r8d, 68(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	56(%rsp), %rcx
	callq	"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %rdx
	callq	"??8error_category@std@@QEBA_NAEBV01@@Z"
	movb	%al, %cl
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, 47(%rsp)                   # 1-byte Spill
	jne	.LBB270_1
	jmp	.LBB270_2
.LBB270_1:
	movq	56(%rsp), %rcx
	callq	"?value@error_code@std@@QEBAHXZ"
	cmpl	68(%rsp), %eax
	sete	%al
	movb	%al, 47(%rsp)                   # 1-byte Spill
.LBB270_2:
	movb	47(%rsp), %al                   # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"
	.globl	"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z" # -- Begin function ?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z
	.p2align	4, 0x90
"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z": # @"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"
.seh_proc "?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 80(%rsp)
	movl	%edx, 76(%rsp)
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rcx
	movq	80(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movl	76(%rsp), %r8d
	movq	(%rcx), %rax
	leaq	48(%rsp), %rdx
	callq	*24(%rax)
	movq	40(%rsp), %rdx                  # 8-byte Reload
	leaq	48(%rsp), %rcx
	callq	"??8std@@YA_NAEBVerror_condition@0@0@Z"
	andb	$1, %al
	movzbl	%al, %eax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1error_category@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1error_category@std@@UEAA@XZ"
	.globl	"??1error_category@std@@UEAA@XZ" # -- Begin function ??1error_category@std@@UEAA@XZ
	.p2align	4, 0x90
"??1error_category@std@@UEAA@XZ":       # @"??1error_category@std@@UEAA@XZ"
.seh_proc "??1error_category@std@@UEAA@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	popq	%rax
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
	.globl	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z" # -- Begin function ??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z
	.p2align	4, 0x90
"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z": # @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
.Lfunc_begin43:
.seh_proc "??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%r8, -16(%rbp)
	movq	%rdx, -24(%rbp)
	movq	%rcx, -32(%rbp)
	movq	-32(%rbp), %rcx
	movq	%rcx, -48(%rbp)                 # 8-byte Spill
	movb	-40(%rbp), %dl
	callq	"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movq	-16(%rbp), %r8
	movq	-24(%rbp), %rdx
.Ltmp426:
	callq	"??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
.Ltmp427:
	jmp	.LBB273_1
.LBB273_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z")@IMGREL
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z@4HA":
.seh_proc "?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z@4HA"
.LBB273_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end43:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
	.p2align	2
"$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z@4HA"@IMGREL # Action
"$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z":
	.long	.Lfunc_begin43@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp426@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp427@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD_K@Z"
                                        # -- End function
	.def	"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z"
	.globl	"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z" # -- Begin function ??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z
	.p2align	4, 0x90
"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z": # @"??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z"
.seh_proc "??0error_condition@std@@QEAA@HAEBVerror_category@1@@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%r8, 16(%rsp)
	movl	%edx, 12(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	12(%rsp), %ecx
	movl	%ecx, (%rax)
	movq	16(%rsp), %rcx
	movq	%rcx, 8(%rax)
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??8error_category@std@@QEBA_NAEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??8error_category@std@@QEBA_NAEBV01@@Z"
	.globl	"??8error_category@std@@QEBA_NAEBV01@@Z" # -- Begin function ??8error_category@std@@QEBA_NAEBV01@@Z
	.p2align	4, 0x90
"??8error_category@std@@QEBA_NAEBV01@@Z": # @"??8error_category@std@@QEBA_NAEBV01@@Z"
.seh_proc "??8error_category@std@@QEBA_NAEBV01@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	addq	$8, %rcx
	callq	"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rcx
	addq	$8, %rcx
	callq	"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"
	movq	%rax, %rcx
	movq	32(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	sete	%al
	andb	$1, %al
	movzbl	%al, %eax
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?category@error_code@std@@QEBAAEBVerror_category@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"
	.globl	"?category@error_code@std@@QEBAAEBVerror_category@2@XZ" # -- Begin function ?category@error_code@std@@QEBAAEBVerror_category@2@XZ
	.p2align	4, 0x90
"?category@error_code@std@@QEBAAEBVerror_category@2@XZ": # @"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"
.seh_proc "?category@error_code@std@@QEBAAEBVerror_category@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	8(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?value@error_code@std@@QEBAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?value@error_code@std@@QEBAHXZ"
	.globl	"?value@error_code@std@@QEBAHXZ" # -- Begin function ?value@error_code@std@@QEBAHXZ
	.p2align	4, 0x90
"?value@error_code@std@@QEBAHXZ":       # @"?value@error_code@std@@QEBAHXZ"
.seh_proc "?value@error_code@std@@QEBAHXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	(%rax), %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"
	.globl	"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z" # -- Begin function ??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z
	.p2align	4, 0x90
"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z": # @"??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"
.seh_proc "??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??8std@@YA_NAEBVerror_condition@0@0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??8std@@YA_NAEBVerror_condition@0@0@Z"
	.globl	"??8std@@YA_NAEBVerror_condition@0@0@Z" # -- Begin function ??8std@@YA_NAEBVerror_condition@0@0@Z
	.p2align	4, 0x90
"??8std@@YA_NAEBVerror_condition@0@0@Z": # @"??8std@@YA_NAEBVerror_condition@0@0@Z"
.seh_proc "??8std@@YA_NAEBVerror_condition@0@0@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	callq	"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rcx
	callq	"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %rdx
	callq	"??8error_category@std@@QEBA_NAEBV01@@Z"
	movb	%al, %cl
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	testb	$1, %cl
	movb	%al, 55(%rsp)                   # 1-byte Spill
	jne	.LBB279_1
	jmp	.LBB279_2
.LBB279_1:
	movq	56(%rsp), %rcx
	callq	"?value@error_condition@std@@QEBAHXZ"
	movl	%eax, 36(%rsp)                  # 4-byte Spill
	movq	64(%rsp), %rcx
	callq	"?value@error_condition@std@@QEBAHXZ"
	movl	%eax, %ecx
	movl	36(%rsp), %eax                  # 4-byte Reload
	cmpl	%ecx, %eax
	sete	%al
	movb	%al, 55(%rsp)                   # 1-byte Spill
.LBB279_2:
	movb	55(%rsp), %al                   # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"
	.globl	"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ" # -- Begin function ?category@error_condition@std@@QEBAAEBVerror_category@2@XZ
	.p2align	4, 0x90
"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ": # @"?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"
.seh_proc "?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	8(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?value@error_condition@std@@QEBAHXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?value@error_condition@std@@QEBAHXZ"
	.globl	"?value@error_condition@std@@QEBAHXZ" # -- Begin function ?value@error_condition@std@@QEBAHXZ
	.p2align	4, 0x90
"?value@error_condition@std@@QEBAHXZ":  # @"?value@error_condition@std@@QEBAHXZ"
.seh_proc "?value@error_condition@std@@QEBAHXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movl	(%rax), %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
	.globl	"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z" # -- Begin function ??0system_error@std@@QEAA@Verror_code@1@PEBD@Z
	.p2align	4, 0x90
"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z": # @"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
.Lfunc_begin44:
.seh_proc "??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$144, %rsp
	.seh_stackalloc 144
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 8(%rbp)
	movq	%rdx, -88(%rbp)                 # 8-byte Spill
	movq	%r8, (%rbp)
	movq	%rcx, -8(%rbp)
	movq	-8(%rbp), %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	movq	(%rbp), %rdx
	leaq	-40(%rbp), %rcx
	movq	%rcx, -72(%rbp)                 # 8-byte Spill
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@QEBD@Z"
	movq	-88(%rbp), %rdx                 # 8-byte Reload
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	-72(%rbp), %r8                  # 8-byte Reload
	movups	(%rdx), %xmm0
	movaps	%xmm0, -64(%rbp)
.Ltmp428:
	leaq	-64(%rbp), %rdx
	callq	"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
.Ltmp429:
	jmp	.LBB282_1
.LBB282_1:
	leaq	-40(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	-80(%rbp), %rax                 # 8-byte Reload
	leaq	"??_7system_error@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$144, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0system_error@std@@QEAA@Verror_code@1@PEBD@Z")@IMGREL
	.section	.text,"xr",discard,"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0system_error@std@@QEAA@Verror_code@1@PEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0system_error@std@@QEAA@Verror_code@1@PEBD@Z@4HA":
.seh_proc "?dtor$2@?0???0system_error@std@@QEAA@Verror_code@1@PEBD@Z@4HA"
.LBB282_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-40(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end44:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
	.p2align	2
"$cppxdata$??0system_error@std@@QEAA@Verror_code@1@PEBD@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0system_error@std@@QEAA@Verror_code@1@PEBD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0system_error@std@@QEAA@Verror_code@1@PEBD@Z")@IMGREL # IPToStateXData
	.long	136                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0system_error@std@@QEAA@Verror_code@1@PEBD@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0system_error@std@@QEAA@Verror_code@1@PEBD@Z@4HA"@IMGREL # Action
"$ip2state$??0system_error@std@@QEAA@Verror_code@1@PEBD@Z":
	.long	.Lfunc_begin44@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp428@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp429@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0system_error@std@@QEAA@Verror_code@1@PEBD@Z"
                                        # -- End function
	.def	"??_Gfailure@ios_base@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gfailure@ios_base@std@@UEAAPEAXI@Z"
	.globl	"??_Gfailure@ios_base@std@@UEAAPEAXI@Z" # -- Begin function ??_Gfailure@ios_base@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gfailure@ios_base@std@@UEAAPEAXI@Z": # @"??_Gfailure@ios_base@std@@UEAAPEAXI@Z"
.seh_proc "??_Gfailure@ios_base@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1failure@ios_base@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB283_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB283_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
	.globl	"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z" # -- Begin function ??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z
	.p2align	4, 0x90
"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z": # @"??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
.seh_proc "??0_System_error@std@@IEAA@Verror_code@1@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
# %bb.0:
	subq	$152, %rsp
	.seh_stackalloc 152
	.seh_endprologue
	movq	%rdx, 40(%rsp)                  # 8-byte Spill
	movq	%r8, 144(%rsp)
	movq	%rcx, 136(%rsp)
	movq	136(%rsp), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	144(%rsp), %rdx
	leaq	72(%rsp), %rcx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
	movq	40(%rsp), %rdx                  # 8-byte Reload
	movq	(%rdx), %rax
	movq	%rax, 56(%rsp)
	movq	8(%rdx), %rax
	movq	%rax, 64(%rsp)
	leaq	104(%rsp), %rcx
	leaq	56(%rsp), %rdx
	leaq	72(%rsp), %r8
	callq	"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	movq	48(%rsp), %rcx                  # 8-byte Reload
	leaq	104(%rsp), %rdx
	callq	"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
	leaq	104(%rsp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	40(%rsp), %rdx                  # 8-byte Reload
	movq	48(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7_System_error@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	movq	(%rdx), %rcx
	movq	%rcx, 24(%rax)
	movq	8(%rdx), %rcx
	movq	%rcx, 32(%rax)
	addq	$152, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_Gsystem_error@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gsystem_error@std@@UEAAPEAXI@Z"
	.globl	"??_Gsystem_error@std@@UEAAPEAXI@Z" # -- Begin function ??_Gsystem_error@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gsystem_error@std@@UEAAPEAXI@Z":    # @"??_Gsystem_error@std@@UEAAPEAXI@Z"
.seh_proc "??_Gsystem_error@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1system_error@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB285_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB285_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.globl	"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z" # -- Begin function ?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z
	.p2align	4, 0x90
"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z": # @"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
.Lfunc_begin45:
.seh_proc "?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$112, %rsp
	.seh_stackalloc 112
	leaq	112(%rsp), %rbp
	.seh_setframe %rbp, 112
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%r8, -72(%rbp)                  # 8-byte Spill
	movq	%rdx, -80(%rbp)                 # 8-byte Spill
	movq	%rcx, %rax
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	movq	%rax, %rdx
	movq	%rdx, -56(%rbp)                 # 8-byte Spill
	movq	%rax, -16(%rbp)
	callq	"?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB286_3
# %bb.1:
.Ltmp430:
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	leaq	"??_C@_02LMMGGCAJ@?3?5?$AA@"(%rip), %rdx
	callq	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"
.Ltmp431:
	jmp	.LBB286_2
.LBB286_2:
	jmp	.LBB286_3
.LBB286_3:
.Ltmp432:
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	leaq	-48(%rbp), %rdx
	callq	"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp433:
	jmp	.LBB286_4
.LBB286_4:
.Ltmp434:
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	leaq	-48(%rbp), %rdx
	callq	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"
.Ltmp435:
	jmp	.LBB286_5
.LBB286_5:
	leaq	-48(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	-72(%rbp), %rdx                 # 8-byte Reload
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z"
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z")@IMGREL
	.section	.text,"xr",discard,"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.seh_endproc
	.def	"?dtor$6@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA":
.seh_proc "?dtor$6@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA"
.LBB286_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	leaq	-48(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.seh_endproc
	.def	"?dtor$7@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$7@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA":
.seh_proc "?dtor$7@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA"
.LBB286_7:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-72(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end45:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.p2align	2
"$cppxdata$?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z")@IMGREL # IPToStateXData
	.long	104                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z":
	.long	-1                              # ToState
	.long	"?dtor$7@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$6@?0??_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z@4HA"@IMGREL # Action
"$ip2state$?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z":
	.long	.Lfunc_begin45@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp430@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp434@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp435@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
                                        # -- End function
	.def	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
	.globl	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z" # -- Begin function ??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z
	.p2align	4, 0x90
"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z": # @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
.Lfunc_begin46:
.seh_proc "??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$96, %rsp
	.seh_stackalloc 96
	leaq	96(%rsp), %rbp
	.seh_setframe %rbp, 96
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	movq	-16(%rbp), %rcx
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"
	movq	%rax, %rdx
	leaq	-32(%rbp), %rcx
	movq	%rcx, -64(%rbp)                 # 8-byte Spill
	callq	"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"
	movq	-64(%rbp), %r8                  # 8-byte Reload
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movb	-40(%rbp), %dl
	callq	"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"
	movq	-16(%rbp), %rcx
	movq	16(%rcx), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	-48(%rbp), %r8                  # 8-byte Reload
	movq	%rax, %rdx
.Ltmp436:
	callq	"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
.Ltmp437:
	jmp	.LBB287_1
.LBB287_1:
	movq	-56(%rbp), %rax                 # 8-byte Reload
	addq	$96, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z")@IMGREL
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z@4HA":
.seh_proc "?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z@4HA"
.LBB287_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	96(%rdx), %rbp
	.seh_endprologue
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	callq	"??1?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end46:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
	.p2align	2
"$cppxdata$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z")@IMGREL # IPToStateXData
	.long	88                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z@4HA"@IMGREL # Action
"$ip2state$??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z":
	.long	.Lfunc_begin46@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp436@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp437@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@AEBV01@@Z"
                                        # -- End function
	.def	"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
	.globl	"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z" # -- Begin function ??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z
	.p2align	4, 0x90
"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z": # @"??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
.seh_proc "??0runtime_error@std@@QEAA@AEBV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@1@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rcx
	callq	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %rdx
	callq	"??0exception@std@@QEAA@QEBD@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	leaq	"??_7runtime_error@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_G_System_error@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G_System_error@std@@UEAAPEAXI@Z"
	.globl	"??_G_System_error@std@@UEAAPEAXI@Z" # -- Begin function ??_G_System_error@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_G_System_error@std@@UEAAPEAXI@Z":   # @"??_G_System_error@std@@UEAAPEAXI@Z"
.seh_proc "??_G_System_error@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1_System_error@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB289_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB289_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"
	.globl	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z" # -- Begin function ?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z
	.p2align	4, 0x90
"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z": # @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"
.seh_proc "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rcx
	callq	"?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	movq	%rax, %rcx
	callq	"??$_Convert_size@_K_K@std@@YA_K_K@Z"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %r8
	movq	48(%rsp), %rdx
	callq	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.globl	"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ" # -- Begin function ?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ
	.p2align	4, 0x90
"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ": # @"?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.seh_proc "?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rdx, 56(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	%rdx, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	72(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"?category@error_code@std@@QEBAAEBVerror_category@2@XZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	callq	"?value@error_code@std@@QEBAHXZ"
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	56(%rsp), %rdx                  # 8-byte Reload
	movl	%eax, %r8d
	movq	(%rcx), %rax
	callq	*16(%rax)
	movq	64(%rsp), %rax                  # 8-byte Reload
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z"
	.globl	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z" # -- Begin function ??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z
	.p2align	4, 0x90
"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z": # @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z"
.seh_proc "??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@$$QEAV01@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rcx
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, %r8
	movb	48(%rsp), %dl
	callq	"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rdx
	callq	"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	64(%rsp), %rdx
	callq	"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"
	.globl	"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z" # -- Begin function ??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z
	.p2align	4, 0x90
"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z": # @"??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"
.seh_proc "??$?0V?$allocator@D@std@@$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_One_then_variadic_args_t@1@$$QEAV?$allocator@D@1@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	%dl, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	callq	"??0?$_String_val@U?$_Simple_types@D@std@@@std@@QEAA@XZ"
                                        # kill: def $rcx killed $rax
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"
	.globl	"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z" # -- Begin function ?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z
	.p2align	4, 0x90
"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z": # @"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"
.seh_proc "?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"
	.globl	"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z" # -- Begin function ?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z
	.p2align	4, 0x90
"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z": # @"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"
.seh_proc "?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 48(%rsp)
	movq	64(%rsp), %rax
	movq	%rax, 40(%rsp)
	movq	64(%rsp), %rdx
	callq	"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"
	movq	64(%rsp), %rcx
	callq	"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"
	.globl	"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z" # -- Begin function ?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z
	.p2align	4, 0x90
"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z": # @"?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"
.seh_proc "?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"
# %bb.0:
	subq	$32, %rsp
	.seh_stackalloc 32
	.seh_endprologue
	movq	%rdx, 24(%rsp)
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	24(%rsp), %rax
	movq	%rax, (%rsp)
	movq	8(%rsp), %rax
	movq	(%rsp), %rcx
	movq	(%rcx), %rdx
	movq	%rdx, (%rax)
	movq	8(%rcx), %rdx
	movq	%rdx, 8(%rax)
	movq	16(%rcx), %rdx
	movq	%rdx, 16(%rax)
	movq	24(%rcx), %rcx
	movq	%rcx, 24(%rax)
	addq	$32, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	.globl	"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ" # -- Begin function ?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ
	.p2align	4, 0x90
"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ": # @"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
.seh_proc "?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rax
	movq	%rax, 40(%rsp)
	movq	40(%rsp), %rax
	movq	$0, 16(%rax)
	movq	40(%rsp), %rax
	movq	$15, 24(%rax)
	movq	40(%rsp), %rcx
	callq	"?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"
	movb	$0, 39(%rsp)
	movq	40(%rsp), %rcx
	leaq	39(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"
	.globl	"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z" # -- Begin function ?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z
	.p2align	4, 0x90
"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z": # @"?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"
.seh_proc "?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rcx, %rax
	movq	%rcx, 8(%rsp)
	movq	%rdx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
	.globl	"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z" # -- Begin function ??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z
	.p2align	4, 0x90
"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z": # @"??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
.seh_proc "??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
# %bb.0:
	subq	$136, %rsp
	.seh_stackalloc 136
	.seh_endprologue
	movq	%r8, 128(%rsp)
	movq	%rdx, 120(%rsp)
	movq	%rcx, 112(%rsp)
	movq	112(%rsp), %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	movq	%rcx, 104(%rsp)
	movq	128(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, %rcx
	movq	56(%rsp), %rax                  # 8-byte Reload
	cmpq	%rcx, %rax
	jbe	.LBB299_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB299_2:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 96(%rsp)
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rax
	movq	%rax, 88(%rsp)
	movq	104(%rsp), %r8
	leaq	80(%rsp), %rcx
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rdx
	callq	"??0_Fake_proxy_ptr_impl@std@@QEAA@AEBU_Fake_allocator@1@AEBU_Container_base0@1@@Z"
	cmpq	$16, 128(%rsp)
	jae	.LBB299_4
# %bb.3:
	movq	128(%rsp), %rcx
	movq	104(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	104(%rsp), %rax
	movq	$15, 24(%rax)
	movq	120(%rsp), %rdx
	movq	104(%rsp), %rcx
	movl	$16, %r8d
	callq	"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	leaq	80(%rsp), %rcx
	callq	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
	jmp	.LBB299_5
.LBB299_4:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	104(%rsp), %rax
	movq	$15, 24(%rax)
	movq	128(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	%rax, 72(%rsp)
	movq	96(%rsp), %rcx
	movq	72(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 64(%rsp)
	movq	104(%rsp), %rcx
	leaq	64(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
	movq	128(%rsp), %rcx
	movq	104(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	72(%rsp), %rcx
	movq	104(%rsp), %rax
	movq	%rcx, 24(%rax)
	movq	128(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 40(%rsp)                  # 8-byte Spill
	movq	120(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	32(%rsp), %rdx                  # 8-byte Reload
	movq	40(%rsp), %r8                   # 8-byte Reload
	movq	%rax, %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	leaq	80(%rsp), %rcx
	callq	"?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
.LBB299_5:
	nop
	addq	$136, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0exception@std@@QEAA@QEBD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0exception@std@@QEAA@QEBD@Z"
	.globl	"??0exception@std@@QEAA@QEBD@Z" # -- Begin function ??0exception@std@@QEAA@QEBD@Z
	.p2align	4, 0x90
"??0exception@std@@QEAA@QEBD@Z":        # @"??0exception@std@@QEAA@QEBD@Z"
.Lfunc_begin47:
.seh_proc "??0exception@std@@QEAA@QEBD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$80, %rsp
	.seh_stackalloc 80
	leaq	80(%rsp), %rbp
	.seh_setframe %rbp, 80
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rdx, -16(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rax
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	leaq	"??_7exception@std@@6B@"(%rip), %rcx
	movq	%rcx, (%rax)
	movq	%rax, %rdx
	addq	$8, %rdx
	xorps	%xmm0, %xmm0
	movups	%xmm0, 8(%rax)
	movq	-16(%rbp), %rax
	movq	%rax, -40(%rbp)
	movb	$1, -32(%rbp)
.Ltmp438:
	leaq	-40(%rbp), %rcx
	callq	__std_exception_copy
.Ltmp439:
	jmp	.LBB300_1
.LBB300_1:
	movq	-48(%rbp), %rax                 # 8-byte Reload
	addq	$80, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??0exception@std@@QEAA@QEBD@Z")@IMGREL
	.section	.text,"xr",discard,"??0exception@std@@QEAA@QEBD@Z"
	.seh_endproc
	.def	"?dtor$2@?0???0exception@std@@QEAA@QEBD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$2@?0???0exception@std@@QEAA@QEBD@Z@4HA":
.seh_proc "?dtor$2@?0???0exception@std@@QEAA@QEBD@Z@4HA"
.LBB300_2:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	80(%rdx), %rbp
	.seh_endprologue
	callq	__std_terminate
	int3
.Lfunc_end47:
	.seh_handlerdata
	.section	.text,"xr",discard,"??0exception@std@@QEAA@QEBD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??0exception@std@@QEAA@QEBD@Z"
	.p2align	2
"$cppxdata$??0exception@std@@QEAA@QEBD@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$??0exception@std@@QEAA@QEBD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$??0exception@std@@QEAA@QEBD@Z")@IMGREL # IPToStateXData
	.long	72                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??0exception@std@@QEAA@QEBD@Z":
	.long	-1                              # ToState
	.long	"?dtor$2@?0???0exception@std@@QEAA@QEBD@Z@4HA"@IMGREL # Action
"$ip2state$??0exception@std@@QEAA@QEBD@Z":
	.long	.Lfunc_begin47@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp438@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp439@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??0exception@std@@QEAA@QEBD@Z"
                                        # -- End function
	.def	"??_Gruntime_error@std@@UEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_Gruntime_error@std@@UEAAPEAXI@Z"
	.globl	"??_Gruntime_error@std@@UEAAPEAXI@Z" # -- Begin function ??_Gruntime_error@std@@UEAAPEAXI@Z
	.p2align	4, 0x90
"??_Gruntime_error@std@@UEAAPEAXI@Z":   # @"??_Gruntime_error@std@@UEAAPEAXI@Z"
.seh_proc "??_Gruntime_error@std@@UEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1runtime_error@std@@UEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB301_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB301_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1runtime_error@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1runtime_error@std@@UEAA@XZ"
	.globl	"??1runtime_error@std@@UEAA@XZ" # -- Begin function ??1runtime_error@std@@UEAA@XZ
	.p2align	4, 0x90
"??1runtime_error@std@@UEAA@XZ":        # @"??1runtime_error@std@@UEAA@XZ"
.seh_proc "??1runtime_error@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1exception@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1_System_error@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1_System_error@std@@UEAA@XZ"
	.globl	"??1_System_error@std@@UEAA@XZ" # -- Begin function ??1_System_error@std@@UEAA@XZ
	.p2align	4, 0x90
"??1_System_error@std@@UEAA@XZ":        # @"??1_System_error@std@@UEAA@XZ"
.seh_proc "??1_System_error@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1runtime_error@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??1system_error@std@@UEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1system_error@std@@UEAA@XZ"
	.globl	"??1system_error@std@@UEAA@XZ"  # -- Begin function ??1system_error@std@@UEAA@XZ
	.p2align	4, 0x90
"??1system_error@std@@UEAA@XZ":         # @"??1system_error@std@@UEAA@XZ"
.seh_proc "??1system_error@std@@UEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1_System_error@std@@UEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.globl	"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z" # -- Begin function ??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z
	.p2align	4, 0x90
"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z": # @"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
.Lfunc_begin48:
.seh_proc "??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$128, %rsp
	.seh_stackalloc 128
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movq	%rcx, -16(%rbp)
	leaq	-24(%rbp), %rcx
	xorl	%edx, %edx
	callq	"??0_Lockit@std@@QEAA@H@Z"
	movq	"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"(%rip), %rax
	movq	%rax, -32(%rbp)
	leaq	"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"(%rip), %rcx
	callq	"??Bid@locale@std@@QEAA_KXZ"
	movq	%rax, -40(%rbp)
	movq	-16(%rbp), %rcx
	movq	-40(%rbp), %rdx
.Ltmp440:
	callq	"?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
.Ltmp441:
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	jmp	.LBB305_1
.LBB305_1:
	movq	-72(%rbp), %rax                 # 8-byte Reload
	movq	%rax, -48(%rbp)
	cmpq	$0, -48(%rbp)
	jne	.LBB305_12
# %bb.2:
	cmpq	$0, -32(%rbp)
	je	.LBB305_4
# %bb.3:
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	jmp	.LBB305_11
.LBB305_4:
	movq	-16(%rbp), %rdx
.Ltmp442:
	leaq	-32(%rbp), %rcx
	callq	"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Ltmp443:
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	jmp	.LBB305_5
.LBB305_5:
	movq	-80(%rbp), %rax                 # 8-byte Reload
	cmpq	$-1, %rax
	jne	.LBB305_8
# %bb.6:
.Ltmp446:
	callq	"?_Throw_bad_cast@std@@YAXXZ"
.Ltmp447:
	jmp	.LBB305_7
.LBB305_7:
.LBB305_8:
	movq	-32(%rbp), %rax
	movq	%rax, -56(%rbp)
	movq	-56(%rbp), %rdx
	leaq	-64(%rbp), %rcx
	callq	"??$?0U?$default_delete@V_Facet_base@std@@@std@@$0A@@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@PEAV_Facet_base@1@@Z"
	movq	-56(%rbp), %rcx
.Ltmp444:
	callq	"?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"
.Ltmp445:
	jmp	.LBB305_9
.LBB305_9:
	movq	-56(%rbp), %rcx
	movq	(%rcx), %rax
	callq	*8(%rax)
	movq	-32(%rbp), %rax
	movq	%rax, "?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"(%rip)
	movq	-32(%rbp), %rax
	movq	%rax, -48(%rbp)
	leaq	-64(%rbp), %rcx
	callq	"?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	jmp	.LBB305_11
.LBB305_11:
	jmp	.LBB305_12
.LBB305_12:
	movq	-48(%rbp), %rax
	movq	%rax, -88(%rbp)                 # 8-byte Spill
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	movq	-88(%rbp), %rax                 # 8-byte Reload
	addq	$128, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z")@IMGREL
	.section	.text,"xr",discard,"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$10@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$10@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"
.LBB305_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-64(%rbp), %rcx
	callq	"??1?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_endproc
	.def	"?dtor$13@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$13@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA":
.seh_proc "?dtor$13@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"
.LBB305_13:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-24(%rbp), %rcx
	callq	"??1_Lockit@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end48:
	.seh_handlerdata
	.section	.text,"xr",discard,"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.p2align	2
"$cppxdata$??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z")@IMGREL # IPToStateXData
	.long	120                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z":
	.long	-1                              # ToState
	.long	"?dtor$13@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$10@?0???$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z@4HA"@IMGREL # Action
"$ip2state$??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z":
	.long	.Lfunc_begin48@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp440@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp444@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp445@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
                                        # -- End function
	.def	"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
	.globl	"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z" # -- Begin function ?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z
	.p2align	4, 0x90
"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z": # @"?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
.seh_proc "?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
# %bb.0:
	subq	$104, %rsp
	.seh_stackalloc 104
	.seh_endprologue
	movq	%rdx, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movl	152(%rsp), %eax
	movb	144(%rsp), %al
	movq	%rdx, 96(%rsp)
	movq	%r9, 88(%rsp)
	movq	%rcx, 80(%rsp)
	movq	80(%rsp), %rcx
	movl	152(%rsp), %r10d
	movb	144(%rsp), %r11b
	movq	88(%rsp), %r9
	movq	(%r8), %rax
	movq	%rax, 64(%rsp)
	movq	8(%r8), %rax
	movq	%rax, 72(%rsp)
	movq	(%rcx), %rax
	leaq	64(%rsp), %r8
	movb	%r11b, 32(%rsp)
	movl	%r10d, 40(%rsp)
	callq	*72(%rax)
	movq	56(%rsp), %rax                  # 8-byte Reload
	addq	$104, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"
	.globl	"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ" # -- Begin function ?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ
	.p2align	4, 0x90
"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ": # @"?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"
.seh_proc "?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	88(%rax), %al
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z"
	.globl	"??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z" # -- Begin function ??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z
	.p2align	4, 0x90
"??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z": # @"??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z"
.seh_proc "??0?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAA@PEAV?$basic_streambuf@DU?$char_traits@D@std@@@1@@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	$0, (%rax)
	movq	8(%rsp), %rcx
	movq	%rcx, 8(%rax)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	.globl	"?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ" # -- Begin function ?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ
	.p2align	4, 0x90
"?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ": # @"?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
.seh_proc "?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movb	(%rax), %al
	andb	$1, %al
	movzbl	%al, %eax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.globl	"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z" # -- Begin function ?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z
	.p2align	4, 0x90
"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z": # @"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
.Lfunc_begin49:
.seh_proc "?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$176, %rsp
	.seh_stackalloc 176
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 40(%rbp)
	movq	%rdx, 32(%rbp)
	movq	%rcx, 24(%rbp)
	cmpq	$0, 24(%rbp)
	je	.LBB310_9
# %bb.1:
	movq	24(%rbp), %rax
	cmpq	$0, (%rax)
	jne	.LBB310_9
# %bb.2:
	movl	$16, %ecx
	callq	"??2@YAPEAX_K@Z"
	movq	%rax, -96(%rbp)                 # 8-byte Spill
	movb	$1, -81(%rbp)
	movq	32(%rbp), %rcx
	callq	"?_C_str@locale@std@@QEBAPEBDXZ"
	movq	%rax, %rdx
.Ltmp448:
	leaq	-80(%rbp), %rcx
	callq	"??0_Locinfo@std@@QEAA@PEBD@Z"
.Ltmp449:
	jmp	.LBB310_3
.LBB310_3:
.Ltmp450:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	xorl	%eax, %eax
	movl	%eax, %r8d
	leaq	-80(%rbp), %rdx
	callq	"??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
.Ltmp451:
	jmp	.LBB310_4
.LBB310_4:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	movb	$0, -81(%rbp)
	movq	24(%rbp), %rax
	movq	%rcx, (%rax)
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	jmp	.LBB310_9
.LBB310_9:
	movl	$4, %eax
	addq	$176, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL
	.section	.text,"xr",discard,"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$5@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$5@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$5@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB310_5:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	-80(%rbp), %rcx
	callq	"??1_Locinfo@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.def	"?dtor$6@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA":
.seh_proc "?dtor$6@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"
.LBB310_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	128(%rdx), %rbp
	.seh_endprologue
	testb	$1, -81(%rbp)
	jne	.LBB310_7
	jmp	.LBB310_8
.LBB310_7:
	movq	-96(%rbp), %rcx                 # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB310_8:
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end49:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.p2align	2
"$cppxdata$?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z")@IMGREL # IPToStateXData
	.long	168                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	-1                              # ToState
	.long	"?dtor$6@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	"?dtor$5@?0??_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z@4HA"@IMGREL # Action
"$ip2state$?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z":
	.long	.Lfunc_begin49@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp448@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp450@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp451@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
                                        # -- End function
	.def	"??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
	.globl	"??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z" # -- Begin function ??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z
	.p2align	4, 0x90
"??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z": # @"??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
.seh_proc "??0?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEAA@AEBV_Locinfo@1@_K@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%r8, 64(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	64(%rsp), %rdx
	callq	"??0facet@locale@std@@IEAA@_K@Z"
	movq	40(%rsp), %rcx                  # 8-byte Reload
	leaq	"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"(%rip), %rax
	movq	%rax, (%rcx)
	movq	56(%rsp), %rdx
	callq	"?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
	movq	40(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
	.globl	"?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z" # -- Begin function ?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z
	.p2align	4, 0x90
"?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z": # @"?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
.seh_proc "?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
	.globl	"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z" # -- Begin function ??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z
	.p2align	4, 0x90
"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z": # @"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
.seh_proc "??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movl	%edx, 60(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	%rcx, 64(%rsp)
	movl	60(%rsp), %eax
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	callq	"??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
	movl	44(%rsp), %eax                  # 4-byte Reload
	cmpl	$0, %eax
	je	.LBB313_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"??3@YAXPEAX@Z"
.LBB313_2:
	movq	64(%rsp), %rax
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z"
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$208, %rsp
	.seh_stackalloc 208
	.seh_endprologue
	movq	%r8, 64(%rsp)                   # 8-byte Spill
	movq	%rdx, 80(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	264(%rsp), %rax
	movb	256(%rsp), %al
	movq	%rdx, 200(%rsp)
	movq	%r9, 192(%rsp)
	movq	%rcx, 184(%rsp)
	movq	184(%rsp), %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	264(%rsp), %r9
	leaq	112(%rsp), %rcx
	leaq	"??_C@_02BBAHNLBA@?$CFp?$AA@"(%rip), %r8
	movl	$64, %edx
	callq	sprintf_s
	movq	64(%rsp), %r8                   # 8-byte Reload
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	80(%rsp), %rdx                  # 8-byte Reload
	cltq
	leaq	112(%rsp), %r10
	movb	256(%rsp), %r11b
	movq	192(%rsp), %r9
	movq	(%r8), %rsi
	movq	%rsi, 96(%rsp)
	movq	8(%r8), %r8
	movq	%r8, 104(%rsp)
	leaq	96(%rsp), %r8
	movb	%r11b, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	movq	88(%rsp), %rax                  # 8-byte Reload
	addq	$208, %rsp
	popq	%rsi
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z";
	.scl	2;
	.type	32;
	.endef
	.globl	__real@4202a05f20000000         # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z
	.section	.rdata,"dr",discard,__real@4202a05f20000000
	.p2align	3
__real@4202a05f20000000:
	.quad	0x4202a05f20000000              # double 1.0E+10
	.globl	__xmm@7fffffffffffffff7fffffffffffffff
	.section	.rdata,"dr",discard,__xmm@7fffffffffffffff7fffffffffffffff
	.p2align	4
__xmm@7fffffffffffffff7fffffffffffffff:
	.quad	0x7fffffffffffffff              # double NaN
	.quad	0x7fffffffffffffff              # double NaN
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
.Lfunc_begin50:
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$288, %rsp                      # imm = 0x120
	.seh_stackalloc 288
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 152(%rbp)
	movq	%r8, -16(%rbp)                  # 8-byte Spill
	movq	%rdx, -8(%rbp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, (%rbp)                    # 8-byte Spill
	movsd	216(%rbp), %xmm0                # xmm0 = mem[0],zero
	movb	208(%rbp), %al
	movq	%rdx, 144(%rbp)
	movq	%r9, 136(%rbp)
	movq	%rcx, 128(%rbp)
	movq	128(%rbp), %rax
	movq	%rax, 8(%rbp)                   # 8-byte Spill
	leaq	96(%rbp), %rcx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	136(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$12288, %eax                    # imm = 0x3000
	movl	%eax, 84(%rbp)
	cmpl	$8192, 84(%rbp)                 # imm = 0x2000
	sete	%al
	andb	$1, %al
	movb	%al, 83(%rbp)
	cmpl	$12288, 84(%rbp)                # imm = 0x3000
	sete	%al
	andb	$1, %al
	movb	%al, 82(%rbp)
	testb	$1, 82(%rbp)
	je	.LBB315_2
# %bb.1:
	movq	$-1, %rax
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	jmp	.LBB315_3
.LBB315_2:
	movq	136(%rbp), %rcx
	callq	"?precision@ios_base@std@@QEBA_JXZ"
	movq	%rax, -24(%rbp)                 # 8-byte Spill
.LBB315_3:
	movq	-24(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 72(%rbp)
	movl	84(%rbp), %edx
	movq	72(%rbp), %rcx
	callq	"??$_Float_put_desired_precision@O@std@@YAH_JH@Z"
	movl	%eax, 68(%rbp)
	movslq	68(%rbp), %rax
	movq	%rax, 56(%rbp)
	testb	$1, 83(%rbp)
	je	.LBB315_6
# %bb.4:
	movsd	216(%rbp), %xmm0                # xmm0 = mem[0],zero
	movaps	__xmm@7fffffffffffffff7fffffffffffffff(%rip), %xmm1 # xmm1 = [NaN,NaN]
	pand	%xmm1, %xmm0
	movsd	__real@4202a05f20000000(%rip), %xmm1 # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB315_6
# %bb.5:
	movsd	216(%rbp), %xmm0                # xmm0 = mem[0],zero
	leaq	52(%rbp), %rdx
	callq	frexpl
	movl	52(%rbp), %ecx
	callq	abs
	imull	$30103, %eax, %eax              # imm = 0x7597
	movl	$100000, %ecx                   # imm = 0x186A0
	cltd
	idivl	%ecx
	cltq
	addq	56(%rbp), %rax
	movq	%rax, 56(%rbp)
.LBB315_6:
	movq	56(%rbp), %rdx
	addq	$50, %rdx
.Ltmp452:
	xorl	%eax, %eax
	movb	%al, %r8b
	leaq	96(%rbp), %rcx
	callq	"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z"
.Ltmp453:
	jmp	.LBB315_7
.LBB315_7:
	movsd	216(%rbp), %xmm0                # xmm0 = mem[0],zero
	movsd	%xmm0, -56(%rbp)                # 8-byte Spill
	movl	72(%rbp), %eax
	movl	%eax, -32(%rbp)                 # 4-byte Spill
	movq	136(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	8(%rbp), %rcx                   # 8-byte Reload
	movl	%eax, %r9d
	leaq	88(%rbp), %rdx
	movb	$76, %r8b
	callq	"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z"
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	leaq	96(%rbp), %rcx
	movq	%rcx, -64(%rbp)                 # 8-byte Spill
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsd	-56(%rbp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movq	-48(%rbp), %rdx                 # 8-byte Reload
	movq	-40(%rbp), %r8                  # 8-byte Reload
	movl	-32(%rbp), %r9d                 # 4-byte Reload
	movq	%rax, %rcx
.Ltmp454:
	movq	%rsp, %rax
	movsd	%xmm0, 32(%rax)
	callq	sprintf_s
.Ltmp455:
	movl	%eax, -28(%rbp)                 # 4-byte Spill
	jmp	.LBB315_8
.LBB315_8:
	movl	-28(%rbp), %eax                 # 4-byte Reload
	cltq
	movq	%rax, 40(%rbp)
	movq	40(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	leaq	96(%rbp), %rcx
	callq	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	movq	-72(%rbp), %r11                 # 8-byte Reload
	movq	8(%rbp), %rcx                   # 8-byte Reload
	movq	-8(%rbp), %rdx                  # 8-byte Reload
	movq	%rax, %r10
	movq	-16(%rbp), %rax                 # 8-byte Reload
	movb	208(%rbp), %r8b
	movq	136(%rbp), %r9
	movups	(%rax), %xmm0
	movaps	%xmm0, 16(%rbp)
.Ltmp456:
	movq	%rsp, %rax
	movq	%r11, 48(%rax)
	movq	%r10, 40(%rax)
	movb	%r8b, 32(%rax)
	leaq	16(%rbp), %r8
	callq	"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
.Ltmp457:
	jmp	.LBB315_9
.LBB315_9:
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	(%rbp), %rax                    # 8-byte Reload
	addq	$288, %rsp                      # imm = 0x120
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z")@IMGREL
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.seh_endproc
	.def	"?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z@4HA":
.seh_proc "?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z@4HA"
.LBB315_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end50:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.p2align	2
"$cppxdata$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z")@IMGREL # IPToStateXData
	.long	280                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z":
	.long	-1                              # ToState
	.long	"?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z@4HA"@IMGREL # Action
"$ip2state$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z":
	.long	.Lfunc_begin50@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp452@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp457@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
.Lfunc_begin51:
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$304, %rsp                      # imm = 0x130
	.seh_stackalloc 304
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 168(%rbp)
	movq	%r8, (%rbp)                     # 8-byte Spill
	movq	%rdx, 8(%rbp)                   # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 16(%rbp)                  # 8-byte Spill
	movsd	232(%rbp), %xmm0                # xmm0 = mem[0],zero
	movb	224(%rbp), %al
	movq	%rdx, 160(%rbp)
	movq	%r9, 152(%rbp)
	movq	%rcx, 144(%rbp)
	movq	144(%rbp), %rax
	movq	%rax, 24(%rbp)                  # 8-byte Spill
	leaq	112(%rbp), %rcx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	152(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$12288, %eax                    # imm = 0x3000
	movl	%eax, 100(%rbp)
	cmpl	$8192, 100(%rbp)                # imm = 0x2000
	sete	%al
	andb	$1, %al
	movb	%al, 99(%rbp)
	cmpl	$12288, 100(%rbp)               # imm = 0x3000
	sete	%al
	andb	$1, %al
	movb	%al, 98(%rbp)
	testb	$1, 98(%rbp)
	je	.LBB316_2
# %bb.1:
	movq	$-1, %rax
	movq	%rax, -8(%rbp)                  # 8-byte Spill
	jmp	.LBB316_3
.LBB316_2:
	movq	152(%rbp), %rcx
	callq	"?precision@ios_base@std@@QEBA_JXZ"
	movq	%rax, -8(%rbp)                  # 8-byte Spill
.LBB316_3:
	movq	-8(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 88(%rbp)
	movl	100(%rbp), %edx
	movq	88(%rbp), %rcx
	callq	"??$_Float_put_desired_precision@N@std@@YAH_JH@Z"
	movl	%eax, 84(%rbp)
	movslq	84(%rbp), %rax
	movq	%rax, 72(%rbp)
	testb	$1, 99(%rbp)
	je	.LBB316_6
# %bb.4:
	movsd	232(%rbp), %xmm0                # xmm0 = mem[0],zero
	movaps	__xmm@7fffffffffffffff7fffffffffffffff(%rip), %xmm1 # xmm1 = [NaN,NaN]
	pand	%xmm1, %xmm0
	movsd	__real@4202a05f20000000(%rip), %xmm1 # xmm1 = mem[0],zero
	ucomisd	%xmm1, %xmm0
	jbe	.LBB316_6
# %bb.5:
	movsd	232(%rbp), %xmm0                # xmm0 = mem[0],zero
	leaq	68(%rbp), %rdx
	callq	frexp
	movl	68(%rbp), %ecx
	callq	abs
	imull	$30103, %eax, %eax              # imm = 0x7597
	movl	$100000, %ecx                   # imm = 0x186A0
	cltd
	idivl	%ecx
	cltq
	addq	72(%rbp), %rax
	movq	%rax, 72(%rbp)
.LBB316_6:
	movq	72(%rbp), %rdx
	addq	$50, %rdx
.Ltmp458:
	xorl	%eax, %eax
	movb	%al, %r8b
	leaq	112(%rbp), %rcx
	callq	"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z"
.Ltmp459:
	jmp	.LBB316_7
.LBB316_7:
	movsd	232(%rbp), %xmm0                # xmm0 = mem[0],zero
	movsd	%xmm0, -40(%rbp)                # 8-byte Spill
	movl	88(%rbp), %eax
	movl	%eax, -16(%rbp)                 # 4-byte Spill
	movq	152(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	24(%rbp), %rcx                  # 8-byte Reload
	movl	%eax, %r9d
	xorl	%eax, %eax
	movl	%eax, -44(%rbp)                 # 4-byte Spill
	movb	%al, %r8b
	leaq	104(%rbp), %rdx
	callq	"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z"
	movq	%rax, -24(%rbp)                 # 8-byte Spill
	leaq	112(%rbp), %rcx
	movq	%rcx, -56(%rbp)                 # 8-byte Spill
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	-56(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, %rdx
	movl	-44(%rbp), %eax                 # 4-byte Reload
	movq	%rdx, -32(%rbp)                 # 8-byte Spill
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movsd	-40(%rbp), %xmm0                # 8-byte Reload
                                        # xmm0 = mem[0],zero
	movq	-32(%rbp), %rdx                 # 8-byte Reload
	movq	-24(%rbp), %r8                  # 8-byte Reload
	movl	-16(%rbp), %r9d                 # 4-byte Reload
	movq	%rax, %rcx
.Ltmp460:
	movq	%rsp, %rax
	movsd	%xmm0, 32(%rax)
	callq	sprintf_s
.Ltmp461:
	movl	%eax, -12(%rbp)                 # 4-byte Spill
	jmp	.LBB316_8
.LBB316_8:
	movl	-12(%rbp), %eax                 # 4-byte Reload
	cltq
	movq	%rax, 56(%rbp)
	movq	56(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	leaq	112(%rbp), %rcx
	callq	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	movq	-64(%rbp), %r11                 # 8-byte Reload
	movq	24(%rbp), %rcx                  # 8-byte Reload
	movq	8(%rbp), %rdx                   # 8-byte Reload
	movq	%rax, %r10
	movq	(%rbp), %rax                    # 8-byte Reload
	movb	224(%rbp), %r8b
	movq	152(%rbp), %r9
	movups	(%rax), %xmm0
	movaps	%xmm0, 32(%rbp)
.Ltmp462:
	movq	%rsp, %rax
	movq	%r11, 48(%rax)
	movq	%r10, 40(%rax)
	movb	%r8b, 32(%rax)
	leaq	32(%rbp), %r8
	callq	"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
.Ltmp463:
	jmp	.LBB316_9
.LBB316_9:
	leaq	112(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	16(%rbp), %rax                  # 8-byte Reload
	addq	$304, %rsp                      # imm = 0x130
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z")@IMGREL
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
	.seh_endproc
	.def	"?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z@4HA":
.seh_proc "?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z@4HA"
.LBB316_10:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$64, %rsp
	.seh_stackalloc 64
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	112(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$64, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end51:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
	.p2align	2
"$cppxdata$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z")@IMGREL # IPToStateXData
	.long	296                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z":
	.long	-1                              # ToState
	.long	"?dtor$10@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z@4HA"@IMGREL # Action
"$ip2state$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z":
	.long	.Lfunc_begin51@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp458@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp463@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z"
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$224, %rsp
	.seh_stackalloc 224
	.seh_endprologue
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	movq	%rdx, 88(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movq	280(%rsp), %rax
	movb	272(%rsp), %al
	movq	%rdx, 216(%rsp)
	movq	%r9, 208(%rsp)
	movq	%rcx, 200(%rsp)
	movq	200(%rsp), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	280(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	208(%rsp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movl	%eax, %r9d
	leaq	120(%rsp), %rdx
	leaq	"??_C@_02CLHGNPPK@Lu?$AA@"(%rip), %r8
	callq	"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
	movq	64(%rsp), %r9                   # 8-byte Reload
	movq	%rax, %r8
	leaq	128(%rsp), %rcx
	movl	$64, %edx
	callq	sprintf_s
	movq	72(%rsp), %r8                   # 8-byte Reload
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rdx                  # 8-byte Reload
	cltq
	leaq	128(%rsp), %r10
	movb	272(%rsp), %r11b
	movq	208(%rsp), %r9
	movq	(%r8), %rsi
	movq	%rsi, 104(%rsp)
	movq	8(%r8), %r8
	movq	%r8, 112(%rsp)
	leaq	104(%rsp), %r8
	movb	%r11b, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	movq	96(%rsp), %rax                  # 8-byte Reload
	addq	$224, %rsp
	popq	%rsi
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z"
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$224, %rsp
	.seh_stackalloc 224
	.seh_endprologue
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	movq	%rdx, 88(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movq	280(%rsp), %rax
	movb	272(%rsp), %al
	movq	%rdx, 216(%rsp)
	movq	%r9, 208(%rsp)
	movq	%rcx, 200(%rsp)
	movq	200(%rsp), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	280(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	208(%rsp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movl	%eax, %r9d
	leaq	120(%rsp), %rdx
	leaq	"??_C@_02HIKPPMOK@Ld?$AA@"(%rip), %r8
	callq	"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
	movq	64(%rsp), %r9                   # 8-byte Reload
	movq	%rax, %r8
	leaq	128(%rsp), %rcx
	movl	$64, %edx
	callq	sprintf_s
	movq	72(%rsp), %r8                   # 8-byte Reload
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rdx                  # 8-byte Reload
	cltq
	leaq	128(%rsp), %r10
	movb	272(%rsp), %r11b
	movq	208(%rsp), %r9
	movq	(%r8), %rsi
	movq	%rsi, 104(%rsp)
	movq	8(%r8), %r8
	movq	%r8, 112(%rsp)
	leaq	104(%rsp), %r8
	movb	%r11b, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	movq	96(%rsp), %rax                  # 8-byte Reload
	addq	$224, %rsp
	popq	%rsi
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z"
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$224, %rsp
	.seh_stackalloc 224
	.seh_endprologue
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	movq	%rdx, 88(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movl	280(%rsp), %eax
	movb	272(%rsp), %al
	movq	%rdx, 216(%rsp)
	movq	%r9, 208(%rsp)
	movq	%rcx, 200(%rsp)
	movq	200(%rsp), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movl	280(%rsp), %eax
	movl	%eax, 68(%rsp)                  # 4-byte Spill
	movq	208(%rsp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movl	%eax, %r9d
	leaq	122(%rsp), %rdx
	leaq	"??_C@_02BDDLJJBK@lu?$AA@"(%rip), %r8
	callq	"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
	movl	68(%rsp), %r9d                  # 4-byte Reload
	movq	%rax, %r8
	leaq	128(%rsp), %rcx
	movl	$64, %edx
	callq	sprintf_s
	movq	72(%rsp), %r8                   # 8-byte Reload
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rdx                  # 8-byte Reload
	cltq
	leaq	128(%rsp), %r10
	movb	272(%rsp), %r11b
	movq	208(%rsp), %r9
	movq	(%r8), %rsi
	movq	%rsi, 104(%rsp)
	movq	8(%r8), %r8
	movq	%r8, 112(%rsp)
	leaq	104(%rsp), %r8
	movb	%r11b, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	movq	96(%rsp), %rax                  # 8-byte Reload
	addq	$224, %rsp
	popq	%rsi
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
# %bb.0:
	pushq	%rsi
	.seh_pushreg %rsi
	subq	$224, %rsp
	.seh_stackalloc 224
	.seh_endprologue
	movq	%r8, 72(%rsp)                   # 8-byte Spill
	movq	%rdx, 88(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 96(%rsp)                  # 8-byte Spill
	movl	280(%rsp), %eax
	movb	272(%rsp), %al
	movq	%rdx, 216(%rsp)
	movq	%r9, 208(%rsp)
	movq	%rcx, 200(%rsp)
	movq	200(%rsp), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movl	280(%rsp), %eax
	movl	%eax, 68(%rsp)                  # 4-byte Spill
	movq	208(%rsp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movl	%eax, %r9d
	leaq	122(%rsp), %rdx
	leaq	"??_C@_02EAOCLKAK@ld?$AA@"(%rip), %r8
	callq	"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
	movl	68(%rsp), %r9d                  # 4-byte Reload
	movq	%rax, %r8
	leaq	128(%rsp), %rcx
	movl	$64, %edx
	callq	sprintf_s
	movq	72(%rsp), %r8                   # 8-byte Reload
	movq	80(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rdx                  # 8-byte Reload
	cltq
	leaq	128(%rsp), %r10
	movb	272(%rsp), %r11b
	movq	208(%rsp), %r9
	movq	(%r8), %rsi
	movq	%rsi, 104(%rsp)
	movq	8(%r8), %r8
	movq	%r8, 112(%rsp)
	leaq	104(%rsp), %r8
	movb	%r11b, 32(%rsp)
	movq	%r10, 40(%rsp)
	movq	%rax, 48(%rsp)
	callq	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	movq	96(%rsp), %rax                  # 8-byte Reload
	addq	$224, %rsp
	popq	%rsi
	retq
	.seh_endproc
                                        # -- End function
	.def	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
	.globl	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z" # -- Begin function ?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z
	.p2align	4, 0x90
"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z": # @"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
.Lfunc_begin52:
.seh_proc "?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$400, %rsp                      # imm = 0x190
	.seh_stackalloc 400
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 264(%rbp)
	movq	%r8, -32(%rbp)                  # 8-byte Spill
	movq	%rdx, -24(%rbp)                 # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, -16(%rbp)                 # 8-byte Spill
	movb	328(%rbp), %al
	movb	320(%rbp), %r8b
	movq	%rdx, 256(%rbp)
	andb	$1, %al
	movb	%al, 255(%rbp)
	movq	%r9, 240(%rbp)
	movq	%rcx, 232(%rbp)
	movq	232(%rbp), %rax
	movq	%rax, -8(%rbp)                  # 8-byte Spill
	movq	240(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$16384, %eax                    # imm = 0x4000
	cmpl	$0, %eax
	jne	.LBB321_2
# %bb.1:
	movq	-24(%rbp), %rdx                 # 8-byte Reload
	movq	-8(%rbp), %rcx                  # 8-byte Reload
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movb	255(%rbp), %r8b
	andb	$1, %r8b
	movzbl	%r8b, %r10d
	movb	320(%rbp), %r11b
	movq	240(%rbp), %r9
	movq	(%rax), %r8
	movq	%r8, 216(%rbp)
	movq	8(%rax), %rax
	movq	%rax, 224(%rbp)
	movq	(%rcx), %rax
	leaq	216(%rbp), %r8
	movb	%r11b, 32(%rsp)
	movl	%r10d, 40(%rsp)
	callq	*72(%rax)
	jmp	.LBB321_20
.LBB321_2:
	movq	240(%rbp), %rcx
	leaq	192(%rbp), %rdx
	movq	%rdx, -48(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-48(%rbp), %rcx                 # 8-byte Reload
.Ltmp464:
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Ltmp465:
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	jmp	.LBB321_3
.LBB321_3:
	leaq	192(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-40(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 208(%rbp)
	leaq	160(%rbp), %rcx
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	testb	$1, 255(%rbp)
	je	.LBB321_7
# %bb.4:
	movq	208(%rbp), %rcx
.Ltmp468:
	leaq	128(%rbp), %rdx
	callq	"?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp469:
	jmp	.LBB321_5
.LBB321_5:
	leaq	160(%rbp), %rcx
	leaq	128(%rbp), %rdx
	callq	"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z"
	leaq	128(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	jmp	.LBB321_9
.LBB321_7:
	movq	208(%rbp), %rcx
.Ltmp466:
	leaq	96(%rbp), %rdx
	callq	"?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp467:
	jmp	.LBB321_8
.LBB321_8:
	leaq	160(%rbp), %rcx
	leaq	96(%rbp), %rdx
	callq	"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z"
	leaq	96(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
.LBB321_9:
	movq	240(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	cmpq	$0, %rax
	jle	.LBB321_11
# %bb.10:
	movq	240(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	leaq	160(%rbp), %rcx
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, %rcx
	movq	-56(%rbp), %rax                 # 8-byte Reload
	cmpq	%rcx, %rax
	ja	.LBB321_12
.LBB321_11:
	movq	$0, 88(%rbp)
	jmp	.LBB321_13
.LBB321_12:
	movq	240(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	leaq	160(%rbp), %rcx
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, %rcx
	movq	-64(%rbp), %rax                 # 8-byte Reload
	subq	%rcx, %rax
	movq	%rax, 88(%rbp)
.LBB321_13:
	movq	240(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$448, %eax                      # imm = 0x1C0
	cmpl	$64, %eax
	je	.LBB321_16
# %bb.14:
	movq	-8(%rbp), %rcx                  # 8-byte Reload
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	88(%rbp), %rdx
	movb	320(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 48(%rbp)
.Ltmp470:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	72(%rbp), %rdx
	leaq	48(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp471:
	jmp	.LBB321_15
.LBB321_15:
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	72(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	80(%rbp), %rcx
	movq	%rcx, 8(%rax)
	movq	$0, 88(%rbp)
.LBB321_16:
	leaq	160(%rbp), %rcx
	movq	%rcx, -80(%rbp)                 # 8-byte Spill
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	-80(%rbp), %rcx                 # 8-byte Reload
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	callq	"?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	movq	-72(%rbp), %rdx                 # 8-byte Reload
	movq	-8(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 16(%rbp)
.Ltmp472:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	32(%rbp), %rdx
	leaq	16(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp473:
	jmp	.LBB321_17
.LBB321_17:
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movups	32(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	240(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"?width@ios_base@std@@QEAA_J_J@Z"
	movq	-8(%rbp), %rcx                  # 8-byte Reload
	movq	-24(%rbp), %rdx                 # 8-byte Reload
                                        # kill: def $r8 killed $rax
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	88(%rbp), %r8
	movb	320(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, (%rbp)
.Ltmp474:
	movq	%rsp, %rax
	movq	%r8, 32(%rax)
	movq	%rbp, %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp475:
	jmp	.LBB321_18
.LBB321_18:
	leaq	160(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	jmp	.LBB321_20
.LBB321_20:
	movq	-16(%rbp), %rax                 # 8-byte Reload
	addq	$400, %rsp                      # imm = 0x190
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z")@IMGREL
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
	.seh_endproc
	.def	"?dtor$6@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$6@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA":
.seh_proc "?dtor$6@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA"
.LBB321_6:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	192(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
	.seh_endproc
	.def	"?dtor$19@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$19@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA":
.seh_proc "?dtor$19@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA"
.LBB321_19:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	160(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end52:
	.seh_handlerdata
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
	.p2align	2
"$cppxdata$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z":
	.long	429065506                       # MagicNumber
	.long	2                               # MaxState
	.long	("$stateUnwindMap$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	4                               # IPMapEntries
	.long	("$ip2state$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z")@IMGREL # IPToStateXData
	.long	392                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z":
	.long	-1                              # ToState
	.long	"?dtor$6@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$19@?0??do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z@4HA"@IMGREL # Action
"$ip2state$?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z":
	.long	.Lfunc_begin52@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp464@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp468@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp475@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"
                                        # -- End function
	.def	"??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
	.globl	"??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ" # -- Begin function ??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ
	.p2align	4, 0x90
"??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ": # @"??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
.seh_proc "??1?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAA@XZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movq	%rcx, 32(%rsp)
	movq	32(%rsp), %rcx
	callq	"??1facet@locale@std@@MEAA@XZ"
	nop
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.globl	"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z" # -- Begin function ?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z
	.p2align	4, 0x90
"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z": # @"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
.Lfunc_begin53:
.seh_proc "?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$592, %rsp                      # imm = 0x250
	.seh_stackalloc 592
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 456(%rbp)
	movq	%r8, 24(%rbp)                   # 8-byte Spill
	movq	%rdx, 32(%rbp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 40(%rbp)                  # 8-byte Spill
	movq	528(%rbp), %rax
	movq	520(%rbp), %rax
	movb	512(%rbp), %al
	movq	%rdx, 448(%rbp)
	movq	%r9, 440(%rbp)
	movq	%rcx, 432(%rbp)
	movq	432(%rbp), %rax
	movq	%rax, 48(%rbp)                  # 8-byte Spill
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	xorl	%ecx, %ecx
                                        # kill: def $rcx killed $ecx
	cmpq	528(%rbp), %rcx
	movb	%al, 63(%rbp)                   # 1-byte Spill
	jae	.LBB323_4
# %bb.1:
	movq	520(%rbp), %rax
	movsbl	(%rax), %ecx
	movb	$1, %al
	cmpl	$43, %ecx
	movb	%al, 23(%rbp)                   # 1-byte Spill
	je	.LBB323_3
# %bb.2:
	movq	520(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$45, %eax
	sete	%al
	movb	%al, 23(%rbp)                   # 1-byte Spill
.LBB323_3:
	movb	23(%rbp), %al                   # 1-byte Reload
	movb	%al, 63(%rbp)                   # 1-byte Spill
.LBB323_4:
	movb	63(%rbp), %al                   # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
                                        # kill: def $rax killed $eax
	movq	%rax, 424(%rbp)
	movq	440(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$3584, %eax                     # imm = 0xE00
	cmpl	$2048, %eax                     # imm = 0x800
	jne	.LBB323_10
# %bb.5:
	movq	424(%rbp), %rax
	addq	$2, %rax
	cmpq	528(%rbp), %rax
	ja	.LBB323_10
# %bb.6:
	movq	520(%rbp), %rax
	movq	424(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	cmpl	$48, %eax
	jne	.LBB323_10
# %bb.7:
	movq	520(%rbp), %rax
	movq	424(%rbp), %rcx
	movsbl	1(%rax,%rcx), %eax
	cmpl	$120, %eax
	je	.LBB323_9
# %bb.8:
	movq	520(%rbp), %rax
	movq	424(%rbp), %rcx
	movsbl	1(%rax,%rcx), %eax
	cmpl	$88, %eax
	jne	.LBB323_10
.LBB323_9:
	movq	424(%rbp), %rax
	addq	$2, %rax
	movq	%rax, 424(%rbp)
.LBB323_10:
	movq	440(%rbp), %rcx
	leaq	400(%rbp), %rdx
	movq	%rdx, (%rbp)                    # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	(%rbp), %rcx                    # 8-byte Reload
.Ltmp476:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp477:
	movq	%rax, 8(%rbp)                   # 8-byte Spill
	jmp	.LBB323_11
.LBB323_11:
	leaq	400(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	8(%rbp), %rax                   # 8-byte Reload
	movq	%rax, 416(%rbp)
	movq	528(%rbp), %rdx
	xorl	%eax, %eax
	movl	%eax, -12(%rbp)                 # 4-byte Spill
	movb	%al, %r8b
	leaq	368(%rbp), %rcx
	movq	%rcx, -24(%rbp)                 # 8-byte Spill
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
                                        # kill: def $rdx killed $rax
	movl	-12(%rbp), %eax                 # 4-byte Reload
	movq	416(%rbp), %rdx
	movq	%rdx, -8(%rbp)                  # 8-byte Spill
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-8(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	520(%rbp), %rdx
	movq	528(%rbp), %rax
	movq	%rdx, %r8
	addq	%rax, %r8
.Ltmp478:
	callq	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
.Ltmp479:
	jmp	.LBB323_12
.LBB323_12:
	movq	440(%rbp), %rcx
	leaq	344(%rbp), %rdx
	movq	%rdx, -40(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-40(%rbp), %rcx                 # 8-byte Reload
.Ltmp480:
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Ltmp481:
	movq	%rax, -32(%rbp)                 # 8-byte Spill
	jmp	.LBB323_13
.LBB323_13:
	leaq	344(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-32(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 360(%rbp)
	movq	360(%rbp), %rcx
.Ltmp482:
	leaq	312(%rbp), %rdx
	callq	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp483:
	jmp	.LBB323_14
.LBB323_14:
	leaq	312(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	movq	%rax, 304(%rbp)
	movq	304(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$127, %eax
	je	.LBB323_29
# %bb.15:
	movq	304(%rbp), %rax
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB323_29
# %bb.16:
	movq	360(%rbp), %rcx
.Ltmp484:
	callq	"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
.Ltmp485:
	movb	%al, -41(%rbp)                  # 1-byte Spill
	jmp	.LBB323_17
.LBB323_17:
	movb	-41(%rbp), %al                  # 1-byte Reload
	movb	%al, 303(%rbp)
.LBB323_18:                             # =>This Inner Loop Header: Depth=1
	movq	304(%rbp), %rax
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	cmpl	$127, %ecx
	movb	%al, -42(%rbp)                  # 1-byte Spill
	je	.LBB323_21
# %bb.19:                               #   in Loop: Header=BB323_18 Depth=1
	movq	304(%rbp), %rax
	movsbl	(%rax), %edx
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	xorl	%ecx, %ecx
	cmpl	%edx, %ecx
	movb	%al, -42(%rbp)                  # 1-byte Spill
	jge	.LBB323_21
# %bb.20:                               #   in Loop: Header=BB323_18 Depth=1
	movq	304(%rbp), %rax
	movsbq	(%rax), %rax
	movq	528(%rbp), %rcx
	subq	424(%rbp), %rcx
	cmpq	%rcx, %rax
	setb	%al
	movb	%al, -42(%rbp)                  # 1-byte Spill
.LBB323_21:                             #   in Loop: Header=BB323_18 Depth=1
	movb	-42(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB323_22
	jmp	.LBB323_28
.LBB323_22:                             #   in Loop: Header=BB323_18 Depth=1
	movq	304(%rbp), %rax
	movsbq	(%rax), %rcx
	movq	528(%rbp), %rax
	subq	%rcx, %rax
	movq	%rax, 528(%rbp)
	movb	303(%rbp), %r9b
	movq	528(%rbp), %rdx
.Ltmp500:
	leaq	368(%rbp), %rcx
	movl	$1, %r8d
	callq	"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z"
.Ltmp501:
	jmp	.LBB323_23
.LBB323_23:                             #   in Loop: Header=BB323_18 Depth=1
	movq	304(%rbp), %rax
	movsbl	1(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB323_27
# %bb.24:                               #   in Loop: Header=BB323_18 Depth=1
	movq	304(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 304(%rbp)
	jmp	.LBB323_27
.LBB323_27:                             #   in Loop: Header=BB323_18 Depth=1
	jmp	.LBB323_18
.LBB323_28:
	jmp	.LBB323_29
.LBB323_29:
	leaq	368(%rbp), %rcx
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, 528(%rbp)
	movq	440(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	cmpq	$0, %rax
	jle	.LBB323_31
# %bb.30:
	movq	440(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	cmpq	528(%rbp), %rax
	ja	.LBB323_32
.LBB323_31:
	movq	$0, 288(%rbp)
	jmp	.LBB323_33
.LBB323_32:
	movq	440(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	subq	528(%rbp), %rax
	movq	%rax, 288(%rbp)
.LBB323_33:
	movq	440(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$448, %eax                      # imm = 0x1C0
	movl	%eax, 284(%rbp)
	cmpl	$64, 284(%rbp)
	je	.LBB323_38
# %bb.34:
	cmpl	$256, 284(%rbp)                 # imm = 0x100
	je	.LBB323_38
# %bb.35:
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	24(%rbp), %rax                  # 8-byte Reload
	movq	288(%rbp), %rdx
	movb	512(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 240(%rbp)
.Ltmp492:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	264(%rbp), %rdx
	leaq	240(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp493:
	jmp	.LBB323_36
.LBB323_36:
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	264(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	$0, 288(%rbp)
	movq	424(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	368(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 208(%rbp)
.Ltmp494:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	224(%rbp), %rdx
	leaq	208(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp495:
	jmp	.LBB323_37
.LBB323_37:
	movq	24(%rbp), %rax                  # 8-byte Reload
	movq	224(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	232(%rbp), %rcx
	movq	%rcx, 8(%rax)
	jmp	.LBB323_45
.LBB323_38:
	cmpl	$256, 284(%rbp)                 # imm = 0x100
	jne	.LBB323_42
# %bb.39:
	movq	424(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	368(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-64(%rbp), %rdx                 # 8-byte Reload
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 176(%rbp)
.Ltmp488:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	192(%rbp), %rdx
	leaq	176(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp489:
	jmp	.LBB323_40
.LBB323_40:
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	192(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	288(%rbp), %rdx
	movb	512(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 144(%rbp)
.Ltmp490:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	160(%rbp), %rdx
	leaq	144(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp491:
	jmp	.LBB323_41
.LBB323_41:
	movq	24(%rbp), %rax                  # 8-byte Reload
	movq	160(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	168(%rbp), %rcx
	movq	%rcx, 8(%rax)
	movq	$0, 288(%rbp)
	jmp	.LBB323_44
.LBB323_42:
	movq	424(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	368(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-72(%rbp), %rdx                 # 8-byte Reload
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 112(%rbp)
.Ltmp486:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	128(%rbp), %rdx
	leaq	112(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp487:
	jmp	.LBB323_43
.LBB323_43:
	movq	24(%rbp), %rax                  # 8-byte Reload
	movq	128(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	136(%rbp), %rcx
	movq	%rcx, 8(%rax)
.LBB323_44:
	jmp	.LBB323_45
.LBB323_45:
	movq	528(%rbp), %rax
	movq	424(%rbp), %rdx
	subq	%rdx, %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	leaq	368(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 80(%rbp)
.Ltmp496:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	96(%rbp), %rdx
	leaq	80(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp497:
	jmp	.LBB323_46
.LBB323_46:
	movq	24(%rbp), %rax                  # 8-byte Reload
	movups	96(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	440(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"?width@ios_base@std@@QEAA_J_J@Z"
	movq	48(%rbp), %rcx                  # 8-byte Reload
	movq	32(%rbp), %rdx                  # 8-byte Reload
                                        # kill: def $r8 killed $rax
	movq	24(%rbp), %rax                  # 8-byte Reload
	movq	288(%rbp), %r8
	movb	512(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 64(%rbp)
.Ltmp498:
	movq	%rsp, %rax
	movq	%r8, 32(%rax)
	leaq	64(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp499:
	jmp	.LBB323_47
.LBB323_47:
	leaq	312(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	leaq	368(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	40(%rbp), %rax                  # 8-byte Reload
	addq	$592, %rsp                      # imm = 0x250
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z")@IMGREL
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.seh_endproc
	.def	"?dtor$25@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$25@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA":
.seh_proc "?dtor$25@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"
.LBB323_25:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	400(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.seh_endproc
	.def	"?dtor$26@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$26@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA":
.seh_proc "?dtor$26@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"
.LBB323_26:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	344(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.seh_endproc
	.def	"?dtor$48@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$48@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA":
.seh_proc "?dtor$48@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"
.LBB323_48:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	312(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.seh_endproc
	.def	"?dtor$49@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$49@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA":
.seh_proc "?dtor$49@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"
.LBB323_49:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	368(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end53:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.p2align	2
"$cppxdata$?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z":
	.long	429065506                       # MagicNumber
	.long	4                               # MaxState
	.long	("$stateUnwindMap$?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	8                               # IPMapEntries
	.long	("$ip2state$?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z")@IMGREL # IPToStateXData
	.long	584                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z":
	.long	-1                              # ToState
	.long	"?dtor$25@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$49@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$48@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$26@?0??_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z@4HA"@IMGREL # Action
"$ip2state$?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z":
	.long	.Lfunc_begin53@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp476@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp477@IMGREL+1               # IP
	.long	-1                              # ToState
	.long	.Ltmp478@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp480@IMGREL+1               # IP
	.long	3                               # ToState
	.long	.Ltmp482@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp484@IMGREL+1               # IP
	.long	2                               # ToState
	.long	.Ltmp499@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
                                        # -- End function
	.def	sprintf_s;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,sprintf_s
	.globl	sprintf_s                       # -- Begin function sprintf_s
	.p2align	4, 0x90
sprintf_s:                              # @sprintf_s
.seh_proc sprintf_s
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r9, 120(%rsp)
	movq	%r8, 80(%rsp)
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	leaq	120(%rsp), %rax
	movq	%rax, 48(%rsp)
	movq	48(%rsp), %r9
	movq	80(%rsp), %r8
	movq	72(%rsp), %rdx
	movq	64(%rsp), %rcx
	movq	%rsp, %rax
	movq	%r9, 32(%rax)
	xorl	%eax, %eax
	movl	%eax, %r9d
	callq	_vsprintf_s_l
	movl	%eax, 60(%rsp)
	movl	60(%rsp), %eax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z"
	.globl	"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z" # -- Begin function ?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z
	.p2align	4, 0x90
"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z": # @"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z"
.seh_proc "?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z"
# %bb.0:
	subq	$136, %rsp
	.seh_stackalloc 136
	.seh_endprologue
	movb	%r9b, 127(%rsp)
	movq	%r8, 112(%rsp)
	movq	%rdx, 104(%rsp)
	movq	%rcx, 96(%rsp)
	movq	96(%rsp), %rcx
	movq	%rcx, 56(%rsp)                  # 8-byte Spill
	movq	104(%rsp), %rdx
	callq	"?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z"
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movq	16(%rcx), %rax
	movq	%rax, 88(%rsp)
	movq	112(%rsp), %rax
	movq	24(%rcx), %rcx
	subq	88(%rsp), %rcx
	cmpq	%rcx, %rax
	ja	.LBB325_2
# %bb.1:
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movq	88(%rsp), %rax
	addq	112(%rsp), %rax
	movq	%rax, 16(%rcx)
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	movq	%rax, 80(%rsp)
	movq	80(%rsp), %rax
	addq	104(%rsp), %rax
	movq	%rax, 72(%rsp)
	movq	88(%rsp), %r8
	subq	104(%rsp), %r8
	addq	$1, %r8
	movq	72(%rsp), %rdx
	movq	72(%rsp), %rcx
	addq	112(%rsp), %rcx
	callq	"?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	127(%rsp), %r8b
	movq	112(%rsp), %rdx
	movq	72(%rsp), %rcx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
                                        # kill: def $rcx killed $rax
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 128(%rsp)
	jmp	.LBB325_3
.LBB325_2:
	movq	56(%rsp), %rcx                  # 8-byte Reload
	movb	127(%rsp), %al
	movq	112(%rsp), %r10
	movq	104(%rsp), %r9
	movq	112(%rsp), %rdx
	movb	64(%rsp), %r8b
	movq	%r10, 32(%rsp)
	movb	%al, 40(%rsp)
	callq	"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"
	movq	%rax, 128(%rsp)
.LBB325_3:
	movq	128(%rsp), %rax
	addq	$136, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	.globl	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ" # -- Begin function ?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ
	.p2align	4, 0x90
"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ": # @"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
.seh_proc "?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	16(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?width@ios_base@std@@QEBA_JXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?width@ios_base@std@@QEBA_JXZ"
	.globl	"?width@ios_base@std@@QEBA_JXZ" # -- Begin function ?width@ios_base@std@@QEBA_JXZ
	.p2align	4, 0x90
"?width@ios_base@std@@QEBA_JXZ":        # @"?width@ios_base@std@@QEBA_JXZ"
.seh_proc "?width@ios_base@std@@QEBA_JXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	40(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
	.globl	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z" # -- Begin function ?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z
	.p2align	4, 0x90
"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z": # @"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.seh_proc "?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 40(%rsp)                   # 8-byte Spill
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	128(%rsp), %rax
	movq	%rdx, 80(%rsp)
	movb	%r9b, 79(%rsp)
	movq	%rcx, 64(%rsp)
.LBB328_1:                              # =>This Inner Loop Header: Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	128(%rsp), %rax
	jae	.LBB328_4
# %bb.2:                                #   in Loop: Header=BB328_1 Depth=1
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movb	79(%rsp), %al
	movb	%al, 39(%rsp)                   # 1-byte Spill
	callq	"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	movb	39(%rsp), %dl                   # 1-byte Reload
	movq	%rax, %rcx
	callq	"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z"
# %bb.3:                                #   in Loop: Header=BB328_1 Depth=1
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	128(%rsp), %rax
	addq	$-1, %rax
	movq	%rax, 128(%rsp)
	callq	"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	jmp	.LBB328_1
.LBB328_4:
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	40(%rsp), %rdx                  # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
	.globl	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z" # -- Begin function ?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z
	.p2align	4, 0x90
"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z": # @"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.seh_proc "?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%r8, 40(%rsp)                   # 8-byte Spill
	movq	%rdx, 48(%rsp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	128(%rsp), %rax
	movq	%rdx, 80(%rsp)
	movq	%r9, 72(%rsp)
	movq	%rcx, 64(%rsp)
.LBB329_1:                              # =>This Inner Loop Header: Depth=1
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	128(%rsp), %rax
	jae	.LBB329_4
# %bb.2:                                #   in Loop: Header=BB329_1 Depth=1
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %rax
	movb	(%rax), %al
	movb	%al, 39(%rsp)                   # 1-byte Spill
	callq	"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	movb	39(%rsp), %dl                   # 1-byte Reload
	movq	%rax, %rcx
	callq	"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z"
# %bb.3:                                #   in Loop: Header=BB329_1 Depth=1
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	128(%rsp), %rax
	addq	$-1, %rax
	movq	%rax, 128(%rsp)
	callq	"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	movq	72(%rsp), %rax
	addq	$1, %rax
	movq	%rax, 72(%rsp)
	jmp	.LBB329_1
.LBB329_4:
	movq	56(%rsp), %rax                  # 8-byte Reload
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movq	40(%rsp), %rdx                  # 8-byte Reload
	movq	(%rdx), %r8
	movq	%r8, (%rcx)
	movq	8(%rdx), %rdx
	movq	%rdx, 8(%rcx)
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?width@ios_base@std@@QEAA_J_J@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?width@ios_base@std@@QEAA_J_J@Z"
	.globl	"?width@ios_base@std@@QEAA_J_J@Z" # -- Begin function ?width@ios_base@std@@QEAA_J_J@Z
	.p2align	4, 0x90
"?width@ios_base@std@@QEAA_J_J@Z":      # @"?width@ios_base@std@@QEAA_J_J@Z"
.seh_proc "?width@ios_base@std@@QEAA_J_J@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rdx, 16(%rsp)
	movq	%rcx, 8(%rsp)
	movq	8(%rsp), %rax
	movq	40(%rax), %rcx
	movq	%rcx, (%rsp)
	movq	16(%rsp), %rcx
	movq	%rcx, 40(%rax)
	movq	(%rsp), %rax
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z"
	.globl	"?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z" # -- Begin function ?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z
	.p2align	4, 0x90
"?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z": # @"?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z"
.seh_proc "?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rax
	movq	16(%rax), %rax
	cmpq	48(%rsp), %rax
	jae	.LBB331_2
# %bb.1:
	callq	"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"
.LBB331_2:
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"
	.globl	"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z" # -- Begin function ??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z
	.p2align	4, 0x90
"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z": # @"??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"
.seh_proc "??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"
# %bb.0:
	subq	$200, %rsp
	.seh_stackalloc 200
	.seh_endprologue
	movb	248(%rsp), %al
	movq	240(%rsp), %rax
	movb	%r8b, 192(%rsp)
	movq	%r9, 184(%rsp)
	movq	%rdx, 176(%rsp)
	movq	%rcx, 168(%rsp)
	movq	168(%rsp), %rcx
	movq	%rcx, 88(%rsp)                  # 8-byte Spill
	movq	%rcx, 160(%rsp)
	movq	160(%rsp), %rax
	movq	16(%rax), %rax
	movq	%rax, 152(%rsp)
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	subq	152(%rsp), %rax
	cmpq	176(%rsp), %rax
	jae	.LBB332_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB332_2:
	movq	88(%rsp), %rcx                  # 8-byte Reload
	movq	152(%rsp), %rax
	addq	176(%rsp), %rax
	movq	%rax, 144(%rsp)
	movq	160(%rsp), %rax
	movq	24(%rax), %rax
	movq	%rax, 136(%rsp)
	movq	144(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	88(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 128(%rsp)
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 120(%rsp)
	movq	120(%rsp), %rcx
	movq	128(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 112(%rsp)
	movq	160(%rsp), %rcx
	callq	"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	movq	144(%rsp), %rcx
	movq	160(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	128(%rsp), %rcx
	movq	160(%rsp), %rax
	movq	%rcx, 24(%rax)
	movq	112(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, 104(%rsp)
	movl	$16, %eax
	cmpq	136(%rsp), %rax
	ja	.LBB332_4
# %bb.3:
	movq	160(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 96(%rsp)
	movb	248(%rsp), %al
	movb	%al, 87(%rsp)                   # 1-byte Spill
	movq	240(%rsp), %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	184(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	152(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	96(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	56(%rsp), %r9                   # 8-byte Reload
	movq	64(%rsp), %r11                  # 8-byte Reload
	movq	72(%rsp), %r10                  # 8-byte Reload
	movq	%rax, %r8
	movb	87(%rsp), %al                   # 1-byte Reload
	movq	104(%rsp), %rdx
	leaq	192(%rsp), %rcx
	movq	%r11, 32(%rsp)
	movq	%r10, 40(%rsp)
	movb	%al, 48(%rsp)
	callq	"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z"
	movq	120(%rsp), %rcx
	movq	136(%rsp), %r8
	addq	$1, %r8
	movq	96(%rsp), %rdx
	callq	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
	movq	112(%rsp), %rcx
	movq	160(%rsp), %rax
	movq	%rcx, (%rax)
	jmp	.LBB332_5
.LBB332_4:
	movb	248(%rsp), %al
	movq	240(%rsp), %r10
	movq	184(%rsp), %r11
	movq	152(%rsp), %r9
	movq	160(%rsp), %r8
	movq	104(%rsp), %rdx
	leaq	192(%rsp), %rcx
	movq	%r11, 32(%rsp)
	movq	%r10, 40(%rsp)
	movb	%al, 48(%rsp)
	callq	"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z"
	movq	160(%rsp), %rcx
	leaq	112(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
.LBB332_5:
	movq	88(%rsp), %rax                  # 8-byte Reload
	addq	$200, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"
	.globl	"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ" # -- Begin function ?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ
	.p2align	4, 0x90
"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ": # @"?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"
.seh_proc "?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	leaq	"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@"(%rip), %rcx
	callq	"?_Xout_of_range@std@@YAXPEBD@Z"
	int3
	.seh_endproc
                                        # -- End function
	.def	"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z"
	.globl	"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z" # -- Begin function ??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z
	.p2align	4, 0x90
"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z": # @"??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z"
.seh_proc "??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	128(%rsp), %al
	movq	120(%rsp), %rax
	movq	112(%rsp), %rax
	movq	%r9, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	112(%rsp), %r8
	movq	56(%rsp), %rdx
	movq	48(%rsp), %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	128(%rsp), %r8b
	movq	120(%rsp), %rdx
	movq	48(%rsp), %rcx
	addq	112(%rsp), %rcx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	movq	64(%rsp), %r8
	subq	112(%rsp), %r8
	addq	$1, %r8
	movq	56(%rsp), %rdx
	addq	112(%rsp), %rdx
	movq	48(%rsp), %rcx
	addq	112(%rsp), %rcx
	addq	120(%rsp), %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	.globl	"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ" # -- Begin function ??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ
	.p2align	4, 0x90
"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ": # @"??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.seh_proc "??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z"
	.globl	"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z" # -- Begin function ??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z
	.p2align	4, 0x90
"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z": # @"??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z"
.seh_proc "??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	%dl, 71(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movb	$1, %al
	cmpq	$0, 8(%rcx)
	movb	%al, 47(%rsp)                   # 1-byte Spill
	je	.LBB336_2
# %bb.1:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	8(%rax), %rcx
	movb	71(%rsp), %dl
	callq	"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"
	movl	%eax, 52(%rsp)
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, 48(%rsp)
	leaq	48(%rsp), %rcx
	leaq	52(%rsp), %rdx
	callq	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	movb	%al, 47(%rsp)                   # 1-byte Spill
.LBB336_2:
	movb	47(%rsp), %al                   # 1-byte Reload
	testb	$1, %al
	jne	.LBB336_3
	jmp	.LBB336_4
.LBB336_3:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movb	$1, (%rax)
.LBB336_4:
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	.globl	"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ" # -- Begin function ??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ
	.p2align	4, 0x90
"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ": # @"??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
.seh_proc "??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"
	.globl	"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z" # -- Begin function ?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z
	.p2align	4, 0x90
"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z": # @"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"
.seh_proc "?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	%dl, 71(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 48(%rsp)                  # 8-byte Spill
	callq	"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	movq	%rax, %rcx
	xorl	%eax, %eax
                                        # kill: def $rax killed $eax
	cmpq	%rcx, %rax
	jge	.LBB338_2
# %bb.1:
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movb	71(%rsp), %al
	movb	%al, 43(%rsp)                   # 1-byte Spill
	callq	"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	movq	%rax, %rcx
	movb	43(%rsp), %al                   # 1-byte Reload
	movb	%al, (%rcx)
	callq	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	movl	%eax, 44(%rsp)                  # 4-byte Spill
	jmp	.LBB338_3
.LBB338_2:
	leaq	71(%rsp), %rcx
	callq	"?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	movq	48(%rsp), %rcx                  # 8-byte Reload
	movl	%eax, %edx
	movq	(%rcx), %rax
	callq	*24(%rax)
	movl	%eax, 44(%rsp)                  # 4-byte Spill
.LBB338_3:
	movl	44(%rsp), %eax                  # 4-byte Reload
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	.globl	"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ" # -- Begin function ?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ
	.p2align	4, 0x90
"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ": # @"?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
.seh_proc "?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movq	%rcx, 16(%rsp)
	movq	16(%rsp), %rax
	movq	%rax, 8(%rsp)                   # 8-byte Spill
	movq	64(%rax), %rax
	cmpq	$0, (%rax)
	je	.LBB339_2
# %bb.1:
	movq	8(%rsp), %rax                   # 8-byte Reload
	movq	88(%rax), %rax
	movl	(%rax), %eax
	movl	%eax, 4(%rsp)                   # 4-byte Spill
	jmp	.LBB339_3
.LBB339_2:
	xorl	%eax, %eax
	movl	%eax, 4(%rsp)                   # 4-byte Spill
	jmp	.LBB339_3
.LBB339_3:
	movl	4(%rsp), %eax                   # 4-byte Reload
	cltq
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	.globl	"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ" # -- Begin function ?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ
	.p2align	4, 0x90
"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ": # @"?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
.seh_proc "?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	88(%rax), %rcx
	movl	(%rcx), %edx
	addl	$-1, %edx
	movl	%edx, (%rcx)
	movq	64(%rax), %rcx
	movq	(%rcx), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, (%rcx)
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	_vsprintf_s_l;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,_vsprintf_s_l
	.globl	_vsprintf_s_l                   # -- Begin function _vsprintf_s_l
	.p2align	4, 0x90
_vsprintf_s_l:                          # @_vsprintf_s_l
.seh_proc _vsprintf_s_l
# %bb.0:
	subq	$136, %rsp
	.seh_stackalloc 136
	.seh_endprologue
	movq	176(%rsp), %rax
	movq	%r9, 128(%rsp)
	movq	%r8, 120(%rsp)
	movq	%rdx, 112(%rsp)
	movq	%rcx, 104(%rsp)
	movq	176(%rsp), %rax
	movq	%rax, 88(%rsp)                  # 8-byte Spill
	movq	128(%rsp), %rax
	movq	%rax, 80(%rsp)                  # 8-byte Spill
	movq	120(%rsp), %rax
	movq	%rax, 72(%rsp)                  # 8-byte Spill
	movq	112(%rsp), %rax
	movq	%rax, 64(%rsp)                  # 8-byte Spill
	movq	104(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	callq	__local_stdio_printf_options
	movq	56(%rsp), %rdx                  # 8-byte Reload
	movq	64(%rsp), %r8                   # 8-byte Reload
	movq	72(%rsp), %r9                   # 8-byte Reload
	movq	80(%rsp), %r10                  # 8-byte Reload
	movq	%rax, %rcx
	movq	88(%rsp), %rax                  # 8-byte Reload
	movq	(%rcx), %rcx
	movq	%r10, 32(%rsp)
	movq	%rax, 40(%rsp)
	callq	__stdio_common_vsprintf_s
	movl	%eax, 100(%rsp)
	cmpl	$0, 100(%rsp)
	jge	.LBB341_2
# %bb.1:
	movl	$4294967295, %eax               # imm = 0xFFFFFFFF
	movl	%eax, 52(%rsp)                  # 4-byte Spill
	jmp	.LBB341_3
.LBB341_2:
	movl	100(%rsp), %eax
	movl	%eax, 52(%rsp)                  # 4-byte Spill
.LBB341_3:
	movl	52(%rsp), %eax                  # 4-byte Reload
	addq	$136, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	__local_stdio_printf_options;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,__local_stdio_printf_options
	.globl	__local_stdio_printf_options    # -- Begin function __local_stdio_printf_options
	.p2align	4, 0x90
__local_stdio_printf_options:           # @__local_stdio_printf_options
# %bb.0:
	leaq	"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA"(%rip), %rax
	retq
                                        # -- End function
	.def	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	.globl	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ" # -- Begin function ??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ
	.p2align	4, 0x90
"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ": # @"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
.seh_proc "??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movb	40(%rsp), %dl
	callq	"??$?0$$V@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAA@U_Zero_then_variadic_args_t@1@@Z"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	leaq	"?_Fake_alloc@std@@3U_Fake_allocator@1@B"(%rip), %rdx
	callq	"?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?precision@ios_base@std@@QEBA_JXZ";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?precision@ios_base@std@@QEBA_JXZ"
	.globl	"?precision@ios_base@std@@QEBA_JXZ" # -- Begin function ?precision@ios_base@std@@QEBA_JXZ
	.p2align	4, 0x90
"?precision@ios_base@std@@QEBA_JXZ":    # @"?precision@ios_base@std@@QEBA_JXZ"
.seh_proc "?precision@ios_base@std@@QEBA_JXZ"
# %bb.0:
	pushq	%rax
	.seh_stackalloc 8
	.seh_endprologue
	movq	%rcx, (%rsp)
	movq	(%rsp), %rax
	movq	32(%rax), %rax
	popq	%rcx
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Float_put_desired_precision@O@std@@YAH_JH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Float_put_desired_precision@O@std@@YAH_JH@Z"
	.globl	"??$_Float_put_desired_precision@O@std@@YAH_JH@Z" # -- Begin function ??$_Float_put_desired_precision@O@std@@YAH_JH@Z
	.p2align	4, 0x90
"??$_Float_put_desired_precision@O@std@@YAH_JH@Z": # @"??$_Float_put_desired_precision@O@std@@YAH_JH@Z"
.seh_proc "??$_Float_put_desired_precision@O@std@@YAH_JH@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movl	%edx, 16(%rsp)
	movq	%rcx, 8(%rsp)
	cmpl	$12288, 16(%rsp)                # imm = 0x3000
	sete	%al
	andb	$1, %al
	movb	%al, 7(%rsp)
	testb	$1, 7(%rsp)
	je	.LBB345_2
# %bb.1:
	movl	$13, 20(%rsp)
	jmp	.LBB345_9
.LBB345_2:
	cmpq	$0, 8(%rsp)
	jle	.LBB345_4
# %bb.3:
	movq	8(%rsp), %rax
                                        # kill: def $eax killed $eax killed $rax
	movl	%eax, 20(%rsp)
	jmp	.LBB345_9
.LBB345_4:
	cmpq	$0, 8(%rsp)
	jne	.LBB345_8
# %bb.5:
	cmpl	$0, 16(%rsp)
	sete	%al
	andb	$1, %al
	movb	%al, 6(%rsp)
	testb	$1, 6(%rsp)
	je	.LBB345_7
# %bb.6:
	movl	$1, 20(%rsp)
	jmp	.LBB345_9
.LBB345_7:
	movl	$0, 20(%rsp)
	jmp	.LBB345_9
.LBB345_8:
	movl	$6, (%rsp)
	movl	$6, 20(%rsp)
.LBB345_9:
	movl	20(%rsp), %eax
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	frexpl;
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,frexpl
	.globl	frexpl                          # -- Begin function frexpl
	.p2align	4, 0x90
frexpl:                                 # @frexpl
.seh_proc frexpl
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movsd	%xmm0, 40(%rsp)
	movq	48(%rsp), %rdx
	movsd	40(%rsp), %xmm0                 # xmm0 = mem[0],zero
	callq	frexp
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z"
	.globl	"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z" # -- Begin function ?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z
	.p2align	4, 0x90
"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z": # @"?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z"
.seh_proc "?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	%r8b, 71(%rsp)
	movq	%rdx, 56(%rsp)
	movq	%rcx, 48(%rsp)
	movq	48(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, 40(%rsp)
	movq	56(%rsp), %rax
	cmpq	40(%rsp), %rax
	ja	.LBB347_2
# %bb.1:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	56(%rsp), %rdx
	callq	"?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z"
	jmp	.LBB347_3
.LBB347_2:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movb	71(%rsp), %r8b
	movq	56(%rsp), %rdx
	subq	40(%rsp), %rdx
	callq	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z"
.LBB347_3:
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z"
	.globl	"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z" # -- Begin function ?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z
	.p2align	4, 0x90
"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z": # @"?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z"
.seh_proc "?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z"
# %bb.0:
	subq	$40, %rsp
	.seh_stackalloc 40
	.seh_endprologue
	movl	%r9d, 36(%rsp)
	movb	%r8b, 35(%rsp)
	movq	%rdx, 24(%rsp)
	movq	%rcx, 16(%rsp)
	movq	24(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$37, (%rax)
	movl	36(%rsp), %eax
	andl	$32, %eax
	cmpl	$0, %eax
	je	.LBB348_2
# %bb.1:
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$43, (%rax)
.LBB348_2:
	movl	36(%rsp), %eax
	andl	$16, %eax
	cmpl	$0, %eax
	je	.LBB348_4
# %bb.3:
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$35, (%rax)
.LBB348_4:
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$46, (%rax)
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$42, (%rax)
	movsbl	35(%rsp), %eax
	cmpl	$0, %eax
	je	.LBB348_6
# %bb.5:
	movb	35(%rsp), %cl
	movq	8(%rsp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 8(%rsp)
	movb	%cl, (%rax)
.LBB348_6:
	movl	36(%rsp), %eax
	andl	$12288, %eax                    # imm = 0x3000
	movl	%eax, (%rsp)
	movl	36(%rsp), %eax
	andl	$4, %eax
	cmpl	$0, %eax
	je	.LBB348_17
# %bb.7:
	cmpl	$8192, (%rsp)                   # imm = 0x2000
	jne	.LBB348_9
# %bb.8:
	movb	$102, 7(%rsp)
	jmp	.LBB348_16
.LBB348_9:
	cmpl	$12288, (%rsp)                  # imm = 0x3000
	jne	.LBB348_11
# %bb.10:
	movb	$65, 7(%rsp)
	jmp	.LBB348_15
.LBB348_11:
	cmpl	$4096, (%rsp)                   # imm = 0x1000
	jne	.LBB348_13
# %bb.12:
	movb	$69, 7(%rsp)
	jmp	.LBB348_14
.LBB348_13:
	movb	$71, 7(%rsp)
.LBB348_14:
	jmp	.LBB348_15
.LBB348_15:
	jmp	.LBB348_16
.LBB348_16:
	jmp	.LBB348_27
.LBB348_17:
	cmpl	$8192, (%rsp)                   # imm = 0x2000
	jne	.LBB348_19
# %bb.18:
	movb	$102, 7(%rsp)
	jmp	.LBB348_26
.LBB348_19:
	cmpl	$12288, (%rsp)                  # imm = 0x3000
	jne	.LBB348_21
# %bb.20:
	movb	$97, 7(%rsp)
	jmp	.LBB348_25
.LBB348_21:
	cmpl	$4096, (%rsp)                   # imm = 0x1000
	jne	.LBB348_23
# %bb.22:
	movb	$101, 7(%rsp)
	jmp	.LBB348_24
.LBB348_23:
	movb	$103, 7(%rsp)
.LBB348_24:
	jmp	.LBB348_25
.LBB348_25:
	jmp	.LBB348_26
.LBB348_26:
	jmp	.LBB348_27
.LBB348_27:
	movb	7(%rsp), %cl
	movq	8(%rsp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 8(%rsp)
	movb	%cl, (%rax)
	movq	8(%rsp), %rax
	movb	$0, (%rax)
	movq	24(%rsp), %rax
	addq	$40, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.globl	"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z" # -- Begin function ?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z
	.p2align	4, 0x90
"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z": # @"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
.Lfunc_begin54:
.seh_proc "?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$640, %rsp                      # imm = 0x280
	.seh_stackalloc 640
	leaq	128(%rsp), %rbp
	.seh_setframe %rbp, 128
	.seh_endprologue
	movq	$-2, 504(%rbp)
	movq	%r8, 40(%rbp)                   # 8-byte Spill
	movq	%rdx, 48(%rbp)                  # 8-byte Spill
	movq	%rdx, %rax
	movq	%rax, 56(%rbp)                  # 8-byte Spill
	movq	576(%rbp), %rax
	movq	568(%rbp), %rax
	movb	560(%rbp), %al
	movq	%rdx, 496(%rbp)
	movq	%r9, 488(%rbp)
	movq	%rcx, 480(%rbp)
	movq	480(%rbp), %rax
	movq	%rax, 64(%rbp)                  # 8-byte Spill
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	xorl	%ecx, %ecx
                                        # kill: def $rcx killed $ecx
	cmpq	576(%rbp), %rcx
	movb	%al, 79(%rbp)                   # 1-byte Spill
	jae	.LBB349_4
# %bb.1:
	movq	568(%rbp), %rax
	movsbl	(%rax), %ecx
	movb	$1, %al
	cmpl	$43, %ecx
	movb	%al, 39(%rbp)                   # 1-byte Spill
	je	.LBB349_3
# %bb.2:
	movq	568(%rbp), %rax
	movsbl	(%rax), %eax
	cmpl	$45, %eax
	sete	%al
	movb	%al, 39(%rbp)                   # 1-byte Spill
.LBB349_3:
	movb	39(%rbp), %al                   # 1-byte Reload
	movb	%al, 79(%rbp)                   # 1-byte Spill
.LBB349_4:
	movb	79(%rbp), %al                   # 1-byte Reload
	andb	$1, %al
	movzbl	%al, %eax
                                        # kill: def $rax killed $eax
	movq	%rax, 472(%rbp)
	movq	488(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$12288, %eax                    # imm = 0x3000
	cmpl	$12288, %eax                    # imm = 0x3000
	je	.LBB349_6
# %bb.5:
	leaq	"??_C@_02MDKMJEGG@eE?$AA@"(%rip), %rax
	movq	%rax, 464(%rbp)
	jmp	.LBB349_12
.LBB349_6:
	leaq	"??_C@_02OOPEBDOJ@pP?$AA@"(%rip), %rax
	movq	%rax, 464(%rbp)
	movq	472(%rbp), %rax
	addq	$2, %rax
	cmpq	576(%rbp), %rax
	ja	.LBB349_11
# %bb.7:
	movq	568(%rbp), %rax
	movq	472(%rbp), %rcx
	movsbl	(%rax,%rcx), %eax
	cmpl	$48, %eax
	jne	.LBB349_11
# %bb.8:
	movq	568(%rbp), %rax
	movq	472(%rbp), %rcx
	movsbl	1(%rax,%rcx), %eax
	cmpl	$120, %eax
	je	.LBB349_10
# %bb.9:
	movq	568(%rbp), %rax
	movq	472(%rbp), %rcx
	movsbl	1(%rax,%rcx), %eax
	cmpl	$88, %eax
	jne	.LBB349_11
.LBB349_10:
	movq	472(%rbp), %rax
	addq	$2, %rax
	movq	%rax, 472(%rbp)
.LBB349_11:
	jmp	.LBB349_12
.LBB349_12:
	movq	464(%rbp), %rdx
	movq	568(%rbp), %rcx
	callq	strcspn
	movq	%rax, 456(%rbp)
	movw	$46, 454(%rbp)
	callq	localeconv
	movq	(%rax), %rax
	movb	(%rax), %al
	movb	%al, 454(%rbp)
	movq	568(%rbp), %rcx
	leaq	454(%rbp), %rdx
	callq	strcspn
	movq	%rax, 440(%rbp)
	movq	488(%rbp), %rcx
	leaq	416(%rbp), %rdx
	movq	%rdx, 16(%rbp)                  # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	16(%rbp), %rcx                  # 8-byte Reload
.Ltmp502:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp503:
	movq	%rax, 24(%rbp)                  # 8-byte Spill
	jmp	.LBB349_13
.LBB349_13:
	leaq	416(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	24(%rbp), %rax                  # 8-byte Reload
	movq	%rax, 432(%rbp)
	movq	576(%rbp), %rdx
	xorl	%eax, %eax
	movl	%eax, 4(%rbp)                   # 4-byte Spill
	movb	%al, %r8b
	leaq	384(%rbp), %rcx
	movq	%rcx, -8(%rbp)                  # 8-byte Spill
	callq	"??0?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@_KD@Z"
	movq	-8(%rbp), %rcx                  # 8-byte Reload
                                        # kill: def $rdx killed $rax
	movl	4(%rbp), %eax                   # 4-byte Reload
	movq	432(%rbp), %rdx
	movq	%rdx, 8(%rbp)                   # 8-byte Spill
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	8(%rbp), %rcx                   # 8-byte Reload
	movq	%rax, %r9
	movq	568(%rbp), %rdx
	movq	576(%rbp), %rax
	movq	%rdx, %r8
	addq	%rax, %r8
.Ltmp504:
	callq	"?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
.Ltmp505:
	jmp	.LBB349_14
.LBB349_14:
	movq	488(%rbp), %rcx
	leaq	360(%rbp), %rdx
	movq	%rdx, -24(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-24(%rbp), %rcx                 # 8-byte Reload
.Ltmp506:
	callq	"??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
.Ltmp507:
	movq	%rax, -16(%rbp)                 # 8-byte Spill
	jmp	.LBB349_15
.LBB349_15:
	leaq	360(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movq	-16(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 376(%rbp)
	movq	376(%rbp), %rcx
.Ltmp508:
	leaq	328(%rbp), %rdx
	callq	"?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
.Ltmp509:
	jmp	.LBB349_16
.LBB349_16:
	movq	376(%rbp), %rcx
.Ltmp510:
	callq	"?thousands_sep@?$numpunct@D@std@@QEBADXZ"
.Ltmp511:
	movb	%al, -25(%rbp)                  # 1-byte Spill
	jmp	.LBB349_17
.LBB349_17:
	movb	-25(%rbp), %al                  # 1-byte Reload
	movb	%al, 327(%rbp)
	movq	440(%rbp), %rax
	cmpq	576(%rbp), %rax
	je	.LBB349_22
# %bb.18:
	movq	376(%rbp), %rcx
.Ltmp512:
	callq	"?decimal_point@?$numpunct@D@std@@QEBADXZ"
.Ltmp513:
	movb	%al, -26(%rbp)                  # 1-byte Spill
	jmp	.LBB349_19
.LBB349_19:
	movq	440(%rbp), %rdx
	leaq	384(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movb	-26(%rbp), %cl                  # 1-byte Reload
	movb	%cl, (%rax)
	jmp	.LBB349_22
.LBB349_22:
	movq	440(%rbp), %rax
	cmpq	576(%rbp), %rax
	jne	.LBB349_24
# %bb.23:
	movq	456(%rbp), %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
	jmp	.LBB349_25
.LBB349_24:
	movq	440(%rbp), %rax
	movq	%rax, -40(%rbp)                 # 8-byte Spill
.LBB349_25:
	movq	-40(%rbp), %rax                 # 8-byte Reload
	movq	%rax, 312(%rbp)
	leaq	328(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	movq	%rax, 304(%rbp)
.LBB349_26:                             # =>This Inner Loop Header: Depth=1
	movq	304(%rbp), %rax
	movsbl	(%rax), %ecx
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	cmpl	$127, %ecx
	movb	%al, -41(%rbp)                  # 1-byte Spill
	je	.LBB349_29
# %bb.27:                               #   in Loop: Header=BB349_26 Depth=1
	movq	304(%rbp), %rax
	movsbl	(%rax), %edx
	xorl	%eax, %eax
                                        # kill: def $al killed $al killed $eax
	xorl	%ecx, %ecx
	cmpl	%edx, %ecx
	movb	%al, -41(%rbp)                  # 1-byte Spill
	jge	.LBB349_29
# %bb.28:                               #   in Loop: Header=BB349_26 Depth=1
	movq	304(%rbp), %rax
	movsbq	(%rax), %rax
	movq	312(%rbp), %rcx
	subq	472(%rbp), %rcx
	cmpq	%rcx, %rax
	setb	%al
	movb	%al, -41(%rbp)                  # 1-byte Spill
.LBB349_29:                             #   in Loop: Header=BB349_26 Depth=1
	movb	-41(%rbp), %al                  # 1-byte Reload
	testb	$1, %al
	jne	.LBB349_30
	jmp	.LBB349_34
.LBB349_30:                             #   in Loop: Header=BB349_26 Depth=1
	movb	327(%rbp), %r9b
	movq	304(%rbp), %rax
	movsbq	(%rax), %rax
	movq	312(%rbp), %rdx
	subq	%rax, %rdx
	movq	%rdx, 312(%rbp)
.Ltmp528:
	leaq	384(%rbp), %rcx
	movl	$1, %r8d
	callq	"?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z"
.Ltmp529:
	jmp	.LBB349_31
.LBB349_31:                             #   in Loop: Header=BB349_26 Depth=1
	movq	304(%rbp), %rax
	movsbl	1(%rax), %ecx
	xorl	%eax, %eax
	cmpl	%ecx, %eax
	jge	.LBB349_33
# %bb.32:                               #   in Loop: Header=BB349_26 Depth=1
	movq	304(%rbp), %rax
	addq	$1, %rax
	movq	%rax, 304(%rbp)
.LBB349_33:                             #   in Loop: Header=BB349_26 Depth=1
	jmp	.LBB349_26
.LBB349_34:
	leaq	384(%rbp), %rcx
	callq	"?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	movq	%rax, 576(%rbp)
	movq	488(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	cmpq	$0, %rax
	jle	.LBB349_36
# %bb.35:
	movq	488(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	cmpq	576(%rbp), %rax
	ja	.LBB349_37
.LBB349_36:
	movq	$0, 296(%rbp)
	jmp	.LBB349_38
.LBB349_37:
	movq	488(%rbp), %rcx
	callq	"?width@ios_base@std@@QEBA_JXZ"
	subq	576(%rbp), %rax
	movq	%rax, 296(%rbp)
.LBB349_38:
	movq	488(%rbp), %rcx
	callq	"?flags@ios_base@std@@QEBAHXZ"
	andl	$448, %eax                      # imm = 0x1C0
	movl	%eax, 292(%rbp)
	cmpl	$64, 292(%rbp)
	je	.LBB349_43
# %bb.39:
	cmpl	$256, 292(%rbp)                 # imm = 0x100
	je	.LBB349_43
# %bb.40:
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	40(%rbp), %rax                  # 8-byte Reload
	movq	296(%rbp), %rdx
	movb	560(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 256(%rbp)
.Ltmp520:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	272(%rbp), %rdx
	leaq	256(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp521:
	jmp	.LBB349_41
.LBB349_41:
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	272(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	$0, 296(%rbp)
	movq	472(%rbp), %rax
	movq	%rax, -56(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	384(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-56(%rbp), %rdx                 # 8-byte Reload
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 224(%rbp)
.Ltmp522:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	240(%rbp), %rdx
	leaq	224(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp523:
	jmp	.LBB349_42
.LBB349_42:
	movq	40(%rbp), %rax                  # 8-byte Reload
	movq	240(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	248(%rbp), %rcx
	movq	%rcx, 8(%rax)
	jmp	.LBB349_50
.LBB349_43:
	cmpl	$256, 292(%rbp)                 # imm = 0x100
	jne	.LBB349_47
# %bb.44:
	movq	472(%rbp), %rax
	movq	%rax, -64(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	384(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-64(%rbp), %rdx                 # 8-byte Reload
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 192(%rbp)
.Ltmp516:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	208(%rbp), %rdx
	leaq	192(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp517:
	jmp	.LBB349_45
.LBB349_45:
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	208(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	296(%rbp), %rdx
	movb	560(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 160(%rbp)
.Ltmp518:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	176(%rbp), %rdx
	leaq	160(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp519:
	jmp	.LBB349_46
.LBB349_46:
	movq	40(%rbp), %rax                  # 8-byte Reload
	movq	176(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	184(%rbp), %rcx
	movq	%rcx, 8(%rax)
	movq	$0, 296(%rbp)
	jmp	.LBB349_49
.LBB349_47:
	movq	472(%rbp), %rax
	movq	%rax, -72(%rbp)                 # 8-byte Spill
	xorl	%eax, %eax
	movl	%eax, %edx
	leaq	384(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-72(%rbp), %rdx                 # 8-byte Reload
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 128(%rbp)
.Ltmp514:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	144(%rbp), %rdx
	leaq	128(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp515:
	jmp	.LBB349_48
.LBB349_48:
	movq	40(%rbp), %rax                  # 8-byte Reload
	movq	144(%rbp), %rcx
	movq	%rcx, (%rax)
	movq	152(%rbp), %rcx
	movq	%rcx, 8(%rax)
.LBB349_49:
	jmp	.LBB349_50
.LBB349_50:
	movq	576(%rbp), %rax
	movq	472(%rbp), %rdx
	subq	%rdx, %rax
	movq	%rax, -80(%rbp)                 # 8-byte Spill
	leaq	384(%rbp), %rcx
	callq	"??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	movq	-80(%rbp), %rdx                 # 8-byte Reload
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	%rax, %r9
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	(%rax), %xmm0
	movaps	%xmm0, 96(%rbp)
.Ltmp524:
	movq	%rsp, %rax
	movq	%rdx, 32(%rax)
	leaq	112(%rbp), %rdx
	leaq	96(%rbp), %r8
	callq	"?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
.Ltmp525:
	jmp	.LBB349_51
.LBB349_51:
	movq	40(%rbp), %rax                  # 8-byte Reload
	movups	112(%rbp), %xmm0
	movups	%xmm0, (%rax)
	movq	488(%rbp), %rcx
	xorl	%eax, %eax
	movl	%eax, %edx
	callq	"?width@ios_base@std@@QEAA_J_J@Z"
	movq	64(%rbp), %rcx                  # 8-byte Reload
	movq	48(%rbp), %rdx                  # 8-byte Reload
                                        # kill: def $r8 killed $rax
	movq	40(%rbp), %rax                  # 8-byte Reload
	movq	296(%rbp), %r8
	movb	560(%rbp), %r9b
	movups	(%rax), %xmm0
	movaps	%xmm0, 80(%rbp)
.Ltmp526:
	movq	%rsp, %rax
	movq	%r8, 32(%rax)
	leaq	80(%rbp), %r8
	callq	"?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
.Ltmp527:
	jmp	.LBB349_52
.LBB349_52:
	leaq	328(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	leaq	384(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	movq	56(%rbp), %rax                  # 8-byte Reload
	addq	$640, %rsp                      # imm = 0x280
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z")@IMGREL
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.seh_endproc
	.def	"?dtor$20@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$20@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA":
.seh_proc "?dtor$20@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"
.LBB349_20:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	416(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.seh_endproc
	.def	"?dtor$21@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$21@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA":
.seh_proc "?dtor$21@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"
.LBB349_21:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	360(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.seh_endproc
	.def	"?dtor$53@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$53@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA":
.seh_proc "?dtor$53@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"
.LBB349_53:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	328(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.seh_endproc
	.def	"?dtor$54@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$54@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA":
.seh_proc "?dtor$54@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"
.LBB349_54:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$48, %rsp
	.seh_stackalloc 48
	leaq	128(%rdx), %rbp
	.seh_endprologue
	leaq	384(%rbp), %rcx
	callq	"??1?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAA@XZ"
	nop
	addq	$48, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end54:
	.seh_handlerdata
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.p2align	2
"$cppxdata$?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z":
	.long	429065506                       # MagicNumber
	.long	4                               # MaxState
	.long	("$stateUnwindMap$?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	8                               # IPMapEntries
	.long	("$ip2state$?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z")@IMGREL # IPToStateXData
	.long	632                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z":
	.long	-1                              # ToState
	.long	"?dtor$20@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"@IMGREL # Action
	.long	-1                              # ToState
	.long	"?dtor$54@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$53@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"@IMGREL # Action
	.long	1                               # ToState
	.long	"?dtor$21@?0??_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z@4HA"@IMGREL # Action
"$ip2state$?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z":
	.long	.Lfunc_begin54@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp502@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp503@IMGREL+1               # IP
	.long	-1                              # ToState
	.long	.Ltmp504@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp506@IMGREL+1               # IP
	.long	3                               # ToState
	.long	.Ltmp508@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp510@IMGREL+1               # IP
	.long	2                               # ToState
	.long	.Ltmp527@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
                                        # -- End function
	.def	"?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z"
	.globl	"?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z" # -- Begin function ?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z
	.p2align	4, 0x90
"?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z": # @"?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z"
.seh_proc "?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movq	%rdx, 64(%rsp)
	movq	%rcx, 56(%rsp)
	movq	56(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movb	$0, 55(%rsp)
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	movq	40(%rsp), %rdx                  # 8-byte Reload
	movq	%rax, %rcx
	movq	64(%rsp), %rax
	movq	%rax, 16(%rdx)
	addq	%rax, %rcx
	leaq	55(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z"
	.globl	"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z" # -- Begin function ?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z
	.p2align	4, 0x90
"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z": # @"?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z"
.seh_proc "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z"
# %bb.0:
	subq	$104, %rsp
	.seh_stackalloc 104
	.seh_endprologue
	movb	%r8b, 95(%rsp)
	movq	%rdx, 80(%rsp)
	movq	%rcx, 72(%rsp)
	movq	72(%rsp), %rcx
	movq	%rcx, 40(%rsp)                  # 8-byte Spill
	movq	16(%rcx), %rax
	movq	%rax, 64(%rsp)
	movq	80(%rsp), %rax
	movq	24(%rcx), %rcx
	subq	64(%rsp), %rcx
	cmpq	%rcx, %rax
	ja	.LBB351_2
# %bb.1:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movq	64(%rsp), %rax
	addq	80(%rsp), %rax
	movq	%rax, 16(%rcx)
	callq	"?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	movq	%rax, 56(%rsp)
	movb	95(%rsp), %r8b
	movq	80(%rsp), %rdx
	movq	56(%rsp), %rcx
	addq	64(%rsp), %rcx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	movb	$0, 55(%rsp)
	movq	56(%rsp), %rcx
	movq	64(%rsp), %rax
	addq	80(%rsp), %rax
	addq	%rax, %rcx
	leaq	55(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	movq	40(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 96(%rsp)
	jmp	.LBB351_3
.LBB351_2:
	movq	40(%rsp), %rcx                  # 8-byte Reload
	movb	95(%rsp), %al
	movq	80(%rsp), %r9
	movq	80(%rsp), %rdx
	movb	48(%rsp), %r8b
	movb	%al, 32(%rsp)
	callq	"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"
	movq	%rax, 96(%rsp)
.LBB351_3:
	movq	96(%rsp), %rax
	addq	$104, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"
	.globl	"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z" # -- Begin function ??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z
	.p2align	4, 0x90
"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z": # @"??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"
.seh_proc "??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"
# %bb.0:
	subq	$184, %rsp
	.seh_stackalloc 184
	.seh_endprologue
	movb	224(%rsp), %al
	movb	%r8b, 176(%rsp)
	movq	%r9, 168(%rsp)
	movq	%rdx, 160(%rsp)
	movq	%rcx, 152(%rsp)
	movq	152(%rsp), %rcx
	movq	%rcx, 72(%rsp)                  # 8-byte Spill
	movq	%rcx, 144(%rsp)
	movq	144(%rsp), %rax
	movq	16(%rax), %rax
	movq	%rax, 136(%rsp)
	callq	"?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	subq	136(%rsp), %rax
	cmpq	160(%rsp), %rax
	jae	.LBB352_2
# %bb.1:
	callq	"?_Xlen_string@std@@YAXXZ"
.LBB352_2:
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	136(%rsp), %rax
	addq	160(%rsp), %rax
	movq	%rax, 128(%rsp)
	movq	144(%rsp), %rax
	movq	24(%rax), %rax
	movq	%rax, 120(%rsp)
	movq	128(%rsp), %rdx
	callq	"?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	movq	72(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 112(%rsp)
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 104(%rsp)
	movq	104(%rsp), %rcx
	movq	112(%rsp), %rdx
	addq	$1, %rdx
	callq	"?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	movq	%rax, 96(%rsp)
	movq	144(%rsp), %rcx
	callq	"?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	movq	128(%rsp), %rcx
	movq	144(%rsp), %rax
	movq	%rcx, 16(%rax)
	movq	112(%rsp), %rcx
	movq	144(%rsp), %rax
	movq	%rcx, 24(%rax)
	movq	96(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	%rax, 88(%rsp)
	movl	$16, %eax
	cmpq	120(%rsp), %rax
	ja	.LBB352_4
# %bb.3:
	movq	144(%rsp), %rax
	movq	(%rax), %rax
	movq	%rax, 80(%rsp)
	movb	224(%rsp), %al
	movb	%al, 71(%rsp)                   # 1-byte Spill
	movq	168(%rsp), %rax
	movq	%rax, 56(%rsp)                  # 8-byte Spill
	movq	136(%rsp), %rax
	movq	%rax, 48(%rsp)                  # 8-byte Spill
	movq	80(%rsp), %rcx
	callq	"??$_Unfancy@D@std@@YAPEADPEAD@Z"
	movq	48(%rsp), %r9                   # 8-byte Reload
	movq	56(%rsp), %r10                  # 8-byte Reload
	movq	%rax, %r8
	movb	71(%rsp), %al                   # 1-byte Reload
	movq	88(%rsp), %rdx
	leaq	176(%rsp), %rcx
	movq	%r10, 32(%rsp)
	movb	%al, 40(%rsp)
	callq	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z"
	movq	104(%rsp), %rcx
	movq	120(%rsp), %r8
	addq	$1, %r8
	movq	80(%rsp), %rdx
	callq	"?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
	movq	96(%rsp), %rcx
	movq	144(%rsp), %rax
	movq	%rcx, (%rax)
	jmp	.LBB352_5
.LBB352_4:
	movb	224(%rsp), %al
	movq	168(%rsp), %r10
	movq	136(%rsp), %r9
	movq	144(%rsp), %r8
	movq	88(%rsp), %rdx
	leaq	176(%rsp), %rcx
	movq	%r10, 32(%rsp)
	movb	%al, 40(%rsp)
	callq	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z"
	movq	144(%rsp), %rcx
	leaq	96(%rsp), %rdx
	callq	"??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
.LBB352_5:
	movq	72(%rsp), %rax                  # 8-byte Reload
	addq	$184, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z"
	.globl	"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z" # -- Begin function ??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z
	.p2align	4, 0x90
"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z": # @"??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z"
.seh_proc "??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z"
# %bb.0:
	subq	$72, %rsp
	.seh_stackalloc 72
	.seh_endprologue
	movb	120(%rsp), %al
	movq	112(%rsp), %rax
	movq	%r9, 64(%rsp)
	movq	%r8, 56(%rsp)
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	64(%rsp), %r8
	movq	56(%rsp), %rdx
	movq	48(%rsp), %rcx
	callq	"?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	movb	120(%rsp), %r8b
	movq	112(%rsp), %rdx
	movq	48(%rsp), %rcx
	addq	64(%rsp), %rcx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	movb	$0, 39(%rsp)
	movq	48(%rsp), %rcx
	movq	64(%rsp), %rax
	addq	112(%rsp), %rax
	addq	%rax, %rcx
	leaq	39(%rsp), %rdx
	callq	"?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	nop
	addq	$72, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Float_put_desired_precision@N@std@@YAH_JH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Float_put_desired_precision@N@std@@YAH_JH@Z"
	.globl	"??$_Float_put_desired_precision@N@std@@YAH_JH@Z" # -- Begin function ??$_Float_put_desired_precision@N@std@@YAH_JH@Z
	.p2align	4, 0x90
"??$_Float_put_desired_precision@N@std@@YAH_JH@Z": # @"??$_Float_put_desired_precision@N@std@@YAH_JH@Z"
.seh_proc "??$_Float_put_desired_precision@N@std@@YAH_JH@Z"
# %bb.0:
	subq	$24, %rsp
	.seh_stackalloc 24
	.seh_endprologue
	movl	%edx, 16(%rsp)
	movq	%rcx, 8(%rsp)
	cmpl	$12288, 16(%rsp)                # imm = 0x3000
	sete	%al
	andb	$1, %al
	movb	%al, 7(%rsp)
	testb	$1, 7(%rsp)
	je	.LBB354_2
# %bb.1:
	movl	$13, 20(%rsp)
	jmp	.LBB354_9
.LBB354_2:
	cmpq	$0, 8(%rsp)
	jle	.LBB354_4
# %bb.3:
	movq	8(%rsp), %rax
                                        # kill: def $eax killed $eax killed $rax
	movl	%eax, 20(%rsp)
	jmp	.LBB354_9
.LBB354_4:
	cmpq	$0, 8(%rsp)
	jne	.LBB354_8
# %bb.5:
	cmpl	$0, 16(%rsp)
	sete	%al
	andb	$1, %al
	movb	%al, 6(%rsp)
	testb	$1, 6(%rsp)
	je	.LBB354_7
# %bb.6:
	movl	$1, 20(%rsp)
	jmp	.LBB354_9
.LBB354_7:
	movl	$0, 20(%rsp)
	jmp	.LBB354_9
.LBB354_8:
	movl	$6, (%rsp)
	movl	$6, 20(%rsp)
.LBB354_9:
	movl	20(%rsp), %eax
	addq	$24, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
	.globl	"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z" # -- Begin function ?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z
	.p2align	4, 0x90
"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z": # @"?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
.seh_proc "?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
# %bb.0:
	subq	$48, %rsp
	.seh_stackalloc 48
	.seh_endprologue
	movl	%r9d, 44(%rsp)
	movq	%r8, 32(%rsp)
	movq	%rdx, 24(%rsp)
	movq	%rcx, 16(%rsp)
	movq	24(%rsp), %rax
	movq	%rax, 8(%rsp)
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$37, (%rax)
	movl	44(%rsp), %eax
	andl	$32, %eax
	cmpl	$0, %eax
	je	.LBB355_2
# %bb.1:
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$43, (%rax)
.LBB355_2:
	movl	44(%rsp), %eax
	andl	$8, %eax
	cmpl	$0, %eax
	je	.LBB355_4
# %bb.3:
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$35, (%rax)
.LBB355_4:
	movq	32(%rsp), %rax
	movsbl	(%rax), %eax
	cmpl	$76, %eax
	je	.LBB355_6
# %bb.5:
	movq	32(%rsp), %rax
	movb	(%rax), %cl
	movq	8(%rsp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 8(%rsp)
	movb	%cl, (%rax)
	jmp	.LBB355_7
.LBB355_6:
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$73, (%rax)
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$54, (%rax)
	movq	8(%rsp), %rax
	movq	%rax, %rcx
	addq	$1, %rcx
	movq	%rcx, 8(%rsp)
	movb	$52, (%rax)
.LBB355_7:
	movl	44(%rsp), %eax
	andl	$3584, %eax                     # imm = 0xE00
	movl	%eax, 4(%rsp)
	cmpl	$1024, 4(%rsp)                  # imm = 0x400
	jne	.LBB355_9
# %bb.8:
	movb	$111, %al
	movb	%al, 3(%rsp)                    # 1-byte Spill
	jmp	.LBB355_13
.LBB355_9:
	cmpl	$2048, 4(%rsp)                  # imm = 0x800
	je	.LBB355_11
# %bb.10:
	movq	32(%rsp), %rax
	movb	1(%rax), %al
	movb	%al, 2(%rsp)                    # 1-byte Spill
	jmp	.LBB355_12
.LBB355_11:
	movl	44(%rsp), %ecx
	andl	$4, %ecx
	movb	$88, %al
	movb	$120, %dl
	movb	%dl, (%rsp)                     # 1-byte Spill
	cmpl	$0, %ecx
	movb	%al, 1(%rsp)                    # 1-byte Spill
	jne	.LBB355_15
# %bb.14:
	movb	(%rsp), %al                     # 1-byte Reload
	movb	%al, 1(%rsp)                    # 1-byte Spill
.LBB355_15:
	movb	1(%rsp), %al                    # 1-byte Reload
	movb	%al, 2(%rsp)                    # 1-byte Spill
.LBB355_12:
	movb	2(%rsp), %al                    # 1-byte Reload
	movb	%al, 3(%rsp)                    # 1-byte Spill
.LBB355_13:
	movb	3(%rsp), %cl                    # 1-byte Reload
	movq	8(%rsp), %rax
	movq	%rax, %rdx
	addq	$1, %rdx
	movq	%rdx, 8(%rsp)
	movb	%cl, (%rax)
	movq	8(%rsp), %rax
	movb	$0, (%rax)
	movq	24(%rsp), %rax
	addq	$48, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z"
	.globl	"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z" # -- Begin function ?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z
	.p2align	4, 0x90
"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z": # @"?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z"
.seh_proc "?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movq	%rdx, 48(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movq	%rcx, 32(%rsp)                  # 8-byte Spill
	movq	48(%rsp), %rdx
	callq	"??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z"
                                        # kill: def $rcx killed $rax
	movq	32(%rsp), %rax                  # 8-byte Reload
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z"
	.globl	"??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z" # -- Begin function ??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z
	.p2align	4, 0x90
"??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z": # @"??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z"
.seh_proc "??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z"
# %bb.0:
	subq	$88, %rsp
	.seh_stackalloc 88
	.seh_endprologue
	movq	%rdx, 72(%rsp)
	movq	%rcx, 64(%rsp)
	movq	64(%rsp), %rax
	movq	%rax, 32(%rsp)                  # 8-byte Spill
	cmpq	72(%rsp), %rax
	jne	.LBB357_2
# %bb.1:
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 80(%rsp)
	jmp	.LBB357_3
.LBB357_2:
	movq	32(%rsp), %rcx                  # 8-byte Reload
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	%rax, 56(%rsp)
	movq	72(%rsp), %rcx
	callq	"?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	%rax, 48(%rsp)
	movl	$0, 44(%rsp)
	callq	"?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	movq	48(%rsp), %rdx
	movq	56(%rsp), %rcx
	callq	"??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z"
	movq	32(%rsp), %rcx                  # 8-byte Reload
	movq	72(%rsp), %rdx
	callq	"?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"
	movq	32(%rsp), %rax                  # 8-byte Reload
	movq	%rax, 80(%rsp)
.LBB357_3:
	movq	80(%rsp), %rax
	addq	$88, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z"
	.globl	"??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z" # -- Begin function ??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z
	.p2align	4, 0x90
"??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z": # @"??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z"
.seh_proc "??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z"
# %bb.0:
	subq	$16, %rsp
	.seh_stackalloc 16
	.seh_endprologue
	movq	%rdx, 8(%rsp)
	movq	%rcx, (%rsp)
	addq	$16, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.def	"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.globl	"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z" # -- Begin function ?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z
	.p2align	4, 0x90
"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z": # @"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
.Lfunc_begin55:
.seh_proc "?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$112, %rsp
	.seh_stackalloc 112
	leaq	112(%rsp), %rbp
	.seh_setframe %rbp, 112
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movb	%dl, -9(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rdx
	movq	%rdx, -64(%rbp)                 # 8-byte Spill
	movl	$0, -28(%rbp)
	leaq	-48(%rbp), %rcx
	callq	"??0sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@AEAV12@@Z"
	leaq	-48(%rbp), %rcx
	callq	"??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	testb	$1, %al
	jne	.LBB359_2
# %bb.1:
	movl	-28(%rbp), %eax
	orl	$4, %eax
	movl	%eax, -28(%rbp)
	jmp	.LBB359_10
.LBB359_2:
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	callq	"?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	movq	%rax, %rcx
	movb	-9(%rbp), %dl
.Ltmp530:
	callq	"?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"
.Ltmp531:
	movl	%eax, -68(%rbp)                 # 4-byte Spill
	jmp	.LBB359_7
.LBB359_5:                              # Block address taken
$ehgcr_359_5:
	jmp	.LBB359_6
.LBB359_6:
	jmp	.LBB359_10
.LBB359_7:
	movl	-68(%rbp), %eax                 # 4-byte Reload
	movl	%eax, -52(%rbp)
	callq	"?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	movl	%eax, -56(%rbp)
	leaq	-56(%rbp), %rcx
	leaq	-52(%rbp), %rdx
	callq	"?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	testb	$1, %al
	jne	.LBB359_8
	jmp	.LBB359_9
.LBB359_8:
	movl	-28(%rbp), %eax
	orl	$4, %eax
	movl	%eax, -28(%rbp)
.LBB359_9:
	jmp	.LBB359_6
.LBB359_10:
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
	movl	-28(%rbp), %edx
.Ltmp534:
	xorl	%eax, %eax
	movb	%al, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp535:
	jmp	.LBB359_11
.LBB359_11:
	leaq	-48(%rbp), %rcx
	callq	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	movq	-64(%rbp), %rax                 # 8-byte Reload
	addq	$112, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z")@IMGREL
	.section	.text,"xr",discard,"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.seh_endproc
	.def	"?catch$3@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?catch$3@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA":
.seh_proc "?catch$3@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA"
	.seh_handler __CxxFrameHandler3, @unwind, @except
.LBB359_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	movq	-64(%rbp), %rcx                 # 8-byte Reload
	movq	(%rcx), %rax
	movslq	4(%rax), %rax
	addq	%rax, %rcx
.Ltmp532:
	movl	$4, %edx
	movb	$1, %r8b
	callq	"?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
.Ltmp533:
	jmp	.LBB359_4
.LBB359_4:
	leaq	.LBB359_5(%rip), %rax
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CATCHRET
	.seh_handlerdata
	.long	("$cppxdata$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z")@IMGREL
	.section	.text,"xr",discard,"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.seh_endproc
	.def	"?dtor$12@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$12@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA":
.seh_proc "?dtor$12@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA"
.LBB359_12:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	112(%rdx), %rbp
	.seh_endprologue
	leaq	-48(%rbp), %rcx
	callq	"??1sentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end55:
	.seh_handlerdata
	.section	.text,"xr",discard,"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.p2align	2
"$cppxdata$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z":
	.long	429065506                       # MagicNumber
	.long	3                               # MaxState
	.long	("$stateUnwindMap$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z")@IMGREL # UnwindMap
	.long	1                               # NumTryBlocks
	.long	("$tryMap$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z")@IMGREL # TryBlockMap
	.long	5                               # IPMapEntries
	.long	("$ip2state$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z")@IMGREL # IPToStateXData
	.long	104                             # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z":
	.long	-1                              # ToState
	.long	"?dtor$12@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA"@IMGREL # Action
	.long	0                               # ToState
	.long	0                               # Action
	.long	0                               # ToState
	.long	0                               # Action
"$tryMap$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z":
	.long	1                               # TryLow
	.long	1                               # TryHigh
	.long	2                               # CatchHigh
	.long	1                               # NumCatches
	.long	("$handlerMap$0$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z")@IMGREL # HandlerArray
"$handlerMap$0$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z":
	.long	64                              # Adjectives
	.long	0                               # Type
	.long	0                               # CatchObjOffset
	.long	"?catch$3@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA"@IMGREL # Handler
	.long	56                              # ParentFrameOffset
"$ip2state$?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z":
	.long	.Lfunc_begin55@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp530@IMGREL+1               # IP
	.long	1                               # ToState
	.long	.Ltmp534@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp535@IMGREL+1               # IP
	.long	-1                              # ToState
	.long	"?catch$3@?0??put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z@4HA"@IMGREL # IP
	.long	2                               # ToState
	.section	.text,"xr",discard,"?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
                                        # -- End function
	.def	"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	.globl	"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z" # -- Begin function ?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z
	.p2align	4, 0x90
"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z": # @"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
.Lfunc_begin56:
.seh_proc "?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	.seh_handler __CxxFrameHandler3, @unwind, @except
# %bb.0:
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$96, %rsp
	.seh_stackalloc 96
	leaq	96(%rsp), %rbp
	.seh_setframe %rbp, 96
	.seh_endprologue
	movq	$-2, -8(%rbp)
	movb	%dl, -9(%rbp)
	movq	%rcx, -24(%rbp)
	movq	-24(%rbp), %rcx
	leaq	-40(%rbp), %rdx
	movq	%rdx, -56(%rbp)                 # 8-byte Spill
	callq	"?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	movq	-56(%rbp), %rcx                 # 8-byte Reload
.Ltmp536:
	callq	"??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
.Ltmp537:
	movq	%rax, -48(%rbp)                 # 8-byte Spill
	jmp	.LBB360_1
.LBB360_1:
	movq	-48(%rbp), %rcx                 # 8-byte Reload
	movb	-9(%rbp), %dl
.Ltmp538:
	callq	"?widen@?$ctype@D@std@@QEBADD@Z"
.Ltmp539:
	movb	%al, -57(%rbp)                  # 1-byte Spill
	jmp	.LBB360_2
.LBB360_2:
	leaq	-40(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	movb	-57(%rbp), %al                  # 1-byte Reload
	addq	$96, %rsp
	popq	%rbp
	retq
	.seh_handlerdata
	.long	("$cppxdata$?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z")@IMGREL
	.section	.text,"xr",discard,"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	.seh_endproc
	.def	"?dtor$3@?0??widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z@4HA";
	.scl	3;
	.type	32;
	.endef
	.p2align	4, 0x90
"?dtor$3@?0??widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z@4HA":
.seh_proc "?dtor$3@?0??widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z@4HA"
.LBB360_3:
	movq	%rdx, 16(%rsp)
	pushq	%rbp
	.seh_pushreg %rbp
	subq	$32, %rsp
	.seh_stackalloc 32
	leaq	96(%rdx), %rbp
	.seh_endprologue
	leaq	-40(%rbp), %rcx
	callq	"??1locale@std@@QEAA@XZ"
	nop
	addq	$32, %rsp
	popq	%rbp
	retq                                    # CLEANUPRET
.Lfunc_end56:
	.seh_handlerdata
	.section	.text,"xr",discard,"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	.seh_endproc
	.section	.xdata,"dr",associative,"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	.p2align	2
"$cppxdata$?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z":
	.long	429065506                       # MagicNumber
	.long	1                               # MaxState
	.long	("$stateUnwindMap$?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z")@IMGREL # UnwindMap
	.long	0                               # NumTryBlocks
	.long	0                               # TryBlockMap
	.long	3                               # IPMapEntries
	.long	("$ip2state$?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z")@IMGREL # IPToStateXData
	.long	88                              # UnwindHelp
	.long	0                               # ESTypeList
	.long	1                               # EHFlags
"$stateUnwindMap$?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z":
	.long	-1                              # ToState
	.long	"?dtor$3@?0??widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z@4HA"@IMGREL # Action
"$ip2state$?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z":
	.long	.Lfunc_begin56@IMGREL           # IP
	.long	-1                              # ToState
	.long	.Ltmp536@IMGREL+1               # IP
	.long	0                               # ToState
	.long	.Ltmp539@IMGREL+1               # IP
	.long	-1                              # ToState
	.section	.text,"xr",discard,"?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
                                        # -- End function
	.def	"?widen@?$ctype@D@std@@QEBADD@Z";
	.scl	2;
	.type	32;
	.endef
	.section	.text,"xr",discard,"?widen@?$ctype@D@std@@QEBADD@Z"
	.globl	"?widen@?$ctype@D@std@@QEBADD@Z" # -- Begin function ?widen@?$ctype@D@std@@QEBADD@Z
	.p2align	4, 0x90
"?widen@?$ctype@D@std@@QEBADD@Z":       # @"?widen@?$ctype@D@std@@QEBADD@Z"
.seh_proc "?widen@?$ctype@D@std@@QEBADD@Z"
# %bb.0:
	subq	$56, %rsp
	.seh_stackalloc 56
	.seh_endprologue
	movb	%dl, 55(%rsp)
	movq	%rcx, 40(%rsp)
	movq	40(%rsp), %rcx
	movb	55(%rsp), %dl
	movq	(%rcx), %rax
	callq	*64(%rax)
	nop
	addq	$56, %rsp
	retq
	.seh_endproc
                                        # -- End function
	.section	.bss,"bw",discard,"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.globl	"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A" # @"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.p2align	3
"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A":
	.zero	8

	.section	.bss,"bw",discard,"?id@?$numpunct@D@std@@2V0locale@2@A"
	.globl	"?id@?$numpunct@D@std@@2V0locale@2@A" # @"?id@?$numpunct@D@std@@2V0locale@2@A"
	.p2align	3
"?id@?$numpunct@D@std@@2V0locale@2@A":
	.zero	8

	.section	.bss,"bw",discard,"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.globl	"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A" # @"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.p2align	3
"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A":
	.zero	8

	.section	.bss,"bw",discard,"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB"
	.globl	"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB" # @"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB"
	.p2align	3
"?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB":
	.quad	0

	.section	.rdata,"dr",discard,"??_C@_00CNPNBAHC@?$AA@"
	.globl	"??_C@_00CNPNBAHC@?$AA@"        # @"??_C@_00CNPNBAHC@?$AA@"
"??_C@_00CNPNBAHC@?$AA@":
	.zero	1

	.section	.rdata,"dr",discard,"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@"
	.globl	"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@" # @"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@"
"??_C@_0BA@ELKIONDK@bad?5locale?5name?$AA@":
	.asciz	"bad locale name"

	.section	.rdata,"dr",largest,"??_7?$ctype@D@std@@6B@"
	.p2align	4                               # @0
.L__unnamed_1:
	.quad	"??_R4?$ctype@D@std@@6B@"
	.quad	"??_G?$ctype@D@std@@MEAAPEAXI@Z"
	.quad	"?_Incref@facet@locale@std@@UEAAXXZ"
	.quad	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
	.quad	"?do_tolower@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
	.quad	"?do_tolower@?$ctype@D@std@@MEBADD@Z"
	.quad	"?do_toupper@?$ctype@D@std@@MEBAPEBDPEADPEBD@Z"
	.quad	"?do_toupper@?$ctype@D@std@@MEBADD@Z"
	.quad	"?do_widen@?$ctype@D@std@@MEBAPEBDPEBD0PEAD@Z"
	.quad	"?do_widen@?$ctype@D@std@@MEBADD@Z"
	.quad	"?do_narrow@?$ctype@D@std@@MEBAPEBDPEBD0DPEAD@Z"
	.quad	"?do_narrow@?$ctype@D@std@@MEBADDD@Z"

	.section	.rdata,"dr",discard,"??_R4?$ctype@D@std@@6B@"
	.globl	"??_R4?$ctype@D@std@@6B@"       # @"??_R4?$ctype@D@std@@6B@"
	.p2align	4
"??_R4?$ctype@D@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV?$ctype@D@std@@@8"@IMGREL
	.long	"??_R3?$ctype@D@std@@8"@IMGREL
	.long	"??_R4?$ctype@D@std@@6B@"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV?$ctype@D@std@@@8"
	.globl	"??_R0?AV?$ctype@D@std@@@8"     # @"??_R0?AV?$ctype@D@std@@@8"
	.p2align	4
"??_R0?AV?$ctype@D@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV?$ctype@D@std@@"
	.zero	4

	.section	.rdata,"dr",discard,"??_R3?$ctype@D@std@@8"
	.globl	"??_R3?$ctype@D@std@@8"         # @"??_R3?$ctype@D@std@@8"
	.p2align	3
"??_R3?$ctype@D@std@@8":
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	5                               # 0x5
	.long	"??_R2?$ctype@D@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2?$ctype@D@std@@8"
	.globl	"??_R2?$ctype@D@std@@8"         # @"??_R2?$ctype@D@std@@8"
	.p2align	4
"??_R2?$ctype@D@std@@8":
	.long	"??_R1A@?0A@EA@?$ctype@D@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@ctype_base@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@facet@locale@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	"??_R17?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@?$ctype@D@std@@8"
	.globl	"??_R1A@?0A@EA@?$ctype@D@std@@8" # @"??_R1A@?0A@EA@?$ctype@D@std@@8"
	.p2align	4
"??_R1A@?0A@EA@?$ctype@D@std@@8":
	.long	"??_R0?AV?$ctype@D@std@@@8"@IMGREL
	.long	4                               # 0x4
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3?$ctype@D@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@ctype_base@std@@8"
	.globl	"??_R1A@?0A@EA@ctype_base@std@@8" # @"??_R1A@?0A@EA@ctype_base@std@@8"
	.p2align	4
"??_R1A@?0A@EA@ctype_base@std@@8":
	.long	"??_R0?AUctype_base@std@@@8"@IMGREL
	.long	3                               # 0x3
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3ctype_base@std@@8"@IMGREL

	.section	.data,"dw",discard,"??_R0?AUctype_base@std@@@8"
	.globl	"??_R0?AUctype_base@std@@@8"    # @"??_R0?AUctype_base@std@@@8"
	.p2align	4
"??_R0?AUctype_base@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AUctype_base@std@@"
	.zero	3

	.section	.rdata,"dr",discard,"??_R3ctype_base@std@@8"
	.globl	"??_R3ctype_base@std@@8"        # @"??_R3ctype_base@std@@8"
	.p2align	3
"??_R3ctype_base@std@@8":
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	4                               # 0x4
	.long	"??_R2ctype_base@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2ctype_base@std@@8"
	.globl	"??_R2ctype_base@std@@8"        # @"??_R2ctype_base@std@@8"
	.p2align	4
"??_R2ctype_base@std@@8":
	.long	"??_R1A@?0A@EA@ctype_base@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@facet@locale@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	"??_R17?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@facet@locale@std@@8"
	.globl	"??_R1A@?0A@EA@facet@locale@std@@8" # @"??_R1A@?0A@EA@facet@locale@std@@8"
	.p2align	4
"??_R1A@?0A@EA@facet@locale@std@@8":
	.long	"??_R0?AVfacet@locale@std@@@8"@IMGREL
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3facet@locale@std@@8"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVfacet@locale@std@@@8"
	.globl	"??_R0?AVfacet@locale@std@@@8"  # @"??_R0?AVfacet@locale@std@@@8"
	.p2align	4
"??_R0?AVfacet@locale@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVfacet@locale@std@@"
	.zero	1

	.section	.rdata,"dr",discard,"??_R3facet@locale@std@@8"
	.globl	"??_R3facet@locale@std@@8"      # @"??_R3facet@locale@std@@8"
	.p2align	3
"??_R3facet@locale@std@@8":
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	3                               # 0x3
	.long	"??_R2facet@locale@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2facet@locale@std@@8"
	.globl	"??_R2facet@locale@std@@8"      # @"??_R2facet@locale@std@@8"
	.p2align	2
"??_R2facet@locale@std@@8":
	.long	"??_R1A@?0A@EA@facet@locale@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	"??_R17?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@_Facet_base@std@@8"
	.globl	"??_R1A@?0A@EA@_Facet_base@std@@8" # @"??_R1A@?0A@EA@_Facet_base@std@@8"
	.p2align	4
"??_R1A@?0A@EA@_Facet_base@std@@8":
	.long	"??_R0?AV_Facet_base@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3_Facet_base@std@@8"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV_Facet_base@std@@@8"
	.globl	"??_R0?AV_Facet_base@std@@@8"   # @"??_R0?AV_Facet_base@std@@@8"
	.p2align	4
"??_R0?AV_Facet_base@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV_Facet_base@std@@"
	.zero	2

	.section	.rdata,"dr",discard,"??_R3_Facet_base@std@@8"
	.globl	"??_R3_Facet_base@std@@8"       # @"??_R3_Facet_base@std@@8"
	.p2align	3
"??_R3_Facet_base@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	"??_R2_Facet_base@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2_Facet_base@std@@8"
	.globl	"??_R2_Facet_base@std@@8"       # @"??_R2_Facet_base@std@@8"
	.p2align	2
"??_R2_Facet_base@std@@8":
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R17?0A@EA@_Crt_new_delete@std@@8"
	.globl	"??_R17?0A@EA@_Crt_new_delete@std@@8" # @"??_R17?0A@EA@_Crt_new_delete@std@@8"
	.p2align	4
"??_R17?0A@EA@_Crt_new_delete@std@@8":
	.long	"??_R0?AU_Crt_new_delete@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	8                               # 0x8
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3_Crt_new_delete@std@@8"@IMGREL

	.section	.data,"dw",discard,"??_R0?AU_Crt_new_delete@std@@@8"
	.globl	"??_R0?AU_Crt_new_delete@std@@@8" # @"??_R0?AU_Crt_new_delete@std@@@8"
	.p2align	4
"??_R0?AU_Crt_new_delete@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AU_Crt_new_delete@std@@"
	.zero	6

	.section	.rdata,"dr",discard,"??_R3_Crt_new_delete@std@@8"
	.globl	"??_R3_Crt_new_delete@std@@8"   # @"??_R3_Crt_new_delete@std@@8"
	.p2align	3
"??_R3_Crt_new_delete@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	"??_R2_Crt_new_delete@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2_Crt_new_delete@std@@8"
	.globl	"??_R2_Crt_new_delete@std@@8"   # @"??_R2_Crt_new_delete@std@@8"
	.p2align	2
"??_R2_Crt_new_delete@std@@8":
	.long	"??_R1A@?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@_Crt_new_delete@std@@8"
	.globl	"??_R1A@?0A@EA@_Crt_new_delete@std@@8" # @"??_R1A@?0A@EA@_Crt_new_delete@std@@8"
	.p2align	4
"??_R1A@?0A@EA@_Crt_new_delete@std@@8":
	.long	"??_R0?AU_Crt_new_delete@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3_Crt_new_delete@std@@8"@IMGREL

	.section	.rdata,"dr",largest,"??_7ctype_base@std@@6B@"
	.p2align	4                               # @1
.L__unnamed_2:
	.quad	"??_R4ctype_base@std@@6B@"
	.quad	"??_Gctype_base@std@@UEAAPEAXI@Z"
	.quad	"?_Incref@facet@locale@std@@UEAAXXZ"
	.quad	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"

	.section	.rdata,"dr",discard,"??_R4ctype_base@std@@6B@"
	.globl	"??_R4ctype_base@std@@6B@"      # @"??_R4ctype_base@std@@6B@"
	.p2align	4
"??_R4ctype_base@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AUctype_base@std@@@8"@IMGREL
	.long	"??_R3ctype_base@std@@8"@IMGREL
	.long	"??_R4ctype_base@std@@6B@"@IMGREL

	.section	.rdata,"dr",largest,"??_7facet@locale@std@@6B@"
	.p2align	4                               # @2
.L__unnamed_3:
	.quad	"??_R4facet@locale@std@@6B@"
	.quad	"??_Gfacet@locale@std@@MEAAPEAXI@Z"
	.quad	"?_Incref@facet@locale@std@@UEAAXXZ"
	.quad	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"

	.section	.rdata,"dr",discard,"??_R4facet@locale@std@@6B@"
	.globl	"??_R4facet@locale@std@@6B@"    # @"??_R4facet@locale@std@@6B@"
	.p2align	4
"??_R4facet@locale@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVfacet@locale@std@@@8"@IMGREL
	.long	"??_R3facet@locale@std@@8"@IMGREL
	.long	"??_R4facet@locale@std@@6B@"@IMGREL

	.section	.rdata,"dr",largest,"??_7_Facet_base@std@@6B@"
	.p2align	4                               # @3
.L__unnamed_4:
	.quad	"??_R4_Facet_base@std@@6B@"
	.quad	"??_G_Facet_base@std@@UEAAPEAXI@Z"
	.quad	_purecall
	.quad	_purecall

	.section	.rdata,"dr",discard,"??_R4_Facet_base@std@@6B@"
	.globl	"??_R4_Facet_base@std@@6B@"     # @"??_R4_Facet_base@std@@6B@"
	.p2align	4
"??_R4_Facet_base@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV_Facet_base@std@@@8"@IMGREL
	.long	"??_R3_Facet_base@std@@8"@IMGREL
	.long	"??_R4_Facet_base@std@@6B@"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVbad_cast@std@@@8"
	.globl	"??_R0?AVbad_cast@std@@@8"      # @"??_R0?AVbad_cast@std@@@8"
	.p2align	4
"??_R0?AVbad_cast@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVbad_cast@std@@"
	.zero	5

	.section	.xdata,"dr",discard,"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24"
	.globl	"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24" # @"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24"
	.p2align	4
"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24":
	.long	0                               # 0x0
	.long	"??_R0?AVbad_cast@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	24                              # 0x18
	.long	"??0bad_cast@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVexception@std@@@8"
	.globl	"??_R0?AVexception@std@@@8"     # @"??_R0?AVexception@std@@@8"
	.p2align	4
"??_R0?AVexception@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVexception@std@@"
	.zero	4

	.section	.xdata,"dr",discard,"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24"
	.globl	"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24" # @"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24"
	.p2align	4
"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24":
	.long	0                               # 0x0
	.long	"??_R0?AVexception@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	24                              # 0x18
	.long	"??0exception@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.xdata,"dr",discard,"_CTA2?AVbad_cast@std@@"
	.globl	"_CTA2?AVbad_cast@std@@"        # @"_CTA2?AVbad_cast@std@@"
	.p2align	3
"_CTA2?AVbad_cast@std@@":
	.long	2                               # 0x2
	.long	"_CT??_R0?AVbad_cast@std@@@8??0bad_cast@std@@QEAA@AEBV01@@Z24"@IMGREL
	.long	"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24"@IMGREL

	.section	.xdata,"dr",discard,"_TI2?AVbad_cast@std@@"
	.globl	"_TI2?AVbad_cast@std@@"         # @"_TI2?AVbad_cast@std@@"
	.p2align	3
"_TI2?AVbad_cast@std@@":
	.long	0                               # 0x0
	.long	"??1bad_cast@std@@UEAA@XZ"@IMGREL
	.long	0                               # 0x0
	.long	"_CTA2?AVbad_cast@std@@"@IMGREL

	.section	.rdata,"dr",discard,"??_C@_08EPJLHIJG@bad?5cast?$AA@"
	.globl	"??_C@_08EPJLHIJG@bad?5cast?$AA@" # @"??_C@_08EPJLHIJG@bad?5cast?$AA@"
"??_C@_08EPJLHIJG@bad?5cast?$AA@":
	.asciz	"bad cast"

	.section	.rdata,"dr",largest,"??_7bad_cast@std@@6B@"
	.p2align	4                               # @4
.L__unnamed_5:
	.quad	"??_R4bad_cast@std@@6B@"
	.quad	"??_Gbad_cast@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4bad_cast@std@@6B@"
	.globl	"??_R4bad_cast@std@@6B@"        # @"??_R4bad_cast@std@@6B@"
	.p2align	4
"??_R4bad_cast@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVbad_cast@std@@@8"@IMGREL
	.long	"??_R3bad_cast@std@@8"@IMGREL
	.long	"??_R4bad_cast@std@@6B@"@IMGREL

	.section	.rdata,"dr",discard,"??_R3bad_cast@std@@8"
	.globl	"??_R3bad_cast@std@@8"          # @"??_R3bad_cast@std@@8"
	.p2align	3
"??_R3bad_cast@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	"??_R2bad_cast@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2bad_cast@std@@8"
	.globl	"??_R2bad_cast@std@@8"          # @"??_R2bad_cast@std@@8"
	.p2align	2
"??_R2bad_cast@std@@8":
	.long	"??_R1A@?0A@EA@bad_cast@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@bad_cast@std@@8"
	.globl	"??_R1A@?0A@EA@bad_cast@std@@8" # @"??_R1A@?0A@EA@bad_cast@std@@8"
	.p2align	4
"??_R1A@?0A@EA@bad_cast@std@@8":
	.long	"??_R0?AVbad_cast@std@@@8"@IMGREL
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3bad_cast@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@exception@std@@8"
	.globl	"??_R1A@?0A@EA@exception@std@@8" # @"??_R1A@?0A@EA@exception@std@@8"
	.p2align	4
"??_R1A@?0A@EA@exception@std@@8":
	.long	"??_R0?AVexception@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3exception@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R3exception@std@@8"
	.globl	"??_R3exception@std@@8"         # @"??_R3exception@std@@8"
	.p2align	3
"??_R3exception@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	"??_R2exception@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2exception@std@@8"
	.globl	"??_R2exception@std@@8"         # @"??_R2exception@std@@8"
	.p2align	2
"??_R2exception@std@@8":
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",largest,"??_7exception@std@@6B@"
	.p2align	4                               # @5
.L__unnamed_6:
	.quad	"??_R4exception@std@@6B@"
	.quad	"??_Gexception@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4exception@std@@6B@"
	.globl	"??_R4exception@std@@6B@"       # @"??_R4exception@std@@6B@"
	.p2align	4
"??_R4exception@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVexception@std@@@8"@IMGREL
	.long	"??_R3exception@std@@8"@IMGREL
	.long	"??_R4exception@std@@6B@"@IMGREL

	.section	.rdata,"dr",discard,"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@"
	.globl	"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@" # @"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@"
"??_C@_0BC@EOODALEL@Unknown?5exception?$AA@":
	.asciz	"Unknown exception"

	.section	.bss,"bw",discard,"?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"
	.globl	"?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB" # @"?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"
	.p2align	3
"?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB":
	.quad	0

	.section	.rdata,"dr",largest,"??_7?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.p2align	4                               # @6
.L__unnamed_7:
	.quad	"??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.quad	"??_G?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
	.quad	"?_Incref@facet@locale@std@@UEAAXXZ"
	.quad	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAPEAX@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAO@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAN@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAM@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_K@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_J@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAK@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAI@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAG@Z"
	.quad	"?do_get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEA_N@Z"

	.section	.rdata,"dr",discard,"??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.globl	"??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" # @"??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.p2align	4
"??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"@IMGREL
	.long	"??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL
	.long	"??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"
	.globl	"??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8" # @"??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"
	.p2align	4
"??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@"
	.zero	6

	.section	.rdata,"dr",discard,"??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.globl	"??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" # @"??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.p2align	3
"??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8":
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	4                               # 0x4
	.long	"??_R2?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.globl	"??_R2?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" # @"??_R2?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.p2align	4
"??_R2?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8":
	.long	"??_R1A@?0A@EA@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@facet@locale@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	"??_R17?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.globl	"??_R1A@?0A@EA@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" # @"??_R1A@?0A@EA@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.p2align	4
"??_R1A@?0A@EA@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8":
	.long	"??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"@IMGREL
	.long	3                               # 0x3
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB"
	.globl	"?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB" # @"?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB"
	.p2align	4
"?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB":
	.asciz	"0123456789ABCDEFabcdef-+Xx"

	.section	.bss,"bw",discard,"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB"
	.globl	"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB" # @"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB"
	.p2align	3
"?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB":
	.quad	0

	.section	.rdata,"dr",largest,"??_7?$numpunct@D@std@@6B@"
	.p2align	4                               # @7
.L__unnamed_8:
	.quad	"??_R4?$numpunct@D@std@@6B@"
	.quad	"??_G?$numpunct@D@std@@MEAAPEAXI@Z"
	.quad	"?_Incref@facet@locale@std@@UEAAXXZ"
	.quad	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
	.quad	"?do_decimal_point@?$numpunct@D@std@@MEBADXZ"
	.quad	"?do_thousands_sep@?$numpunct@D@std@@MEBADXZ"
	.quad	"?do_grouping@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.quad	"?do_falsename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.quad	"?do_truename@?$numpunct@D@std@@MEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"

	.section	.rdata,"dr",discard,"??_R4?$numpunct@D@std@@6B@"
	.globl	"??_R4?$numpunct@D@std@@6B@"    # @"??_R4?$numpunct@D@std@@6B@"
	.p2align	4
"??_R4?$numpunct@D@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV?$numpunct@D@std@@@8"@IMGREL
	.long	"??_R3?$numpunct@D@std@@8"@IMGREL
	.long	"??_R4?$numpunct@D@std@@6B@"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV?$numpunct@D@std@@@8"
	.globl	"??_R0?AV?$numpunct@D@std@@@8"  # @"??_R0?AV?$numpunct@D@std@@@8"
	.p2align	4
"??_R0?AV?$numpunct@D@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV?$numpunct@D@std@@"
	.zero	1

	.section	.rdata,"dr",discard,"??_R3?$numpunct@D@std@@8"
	.globl	"??_R3?$numpunct@D@std@@8"      # @"??_R3?$numpunct@D@std@@8"
	.p2align	3
"??_R3?$numpunct@D@std@@8":
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	4                               # 0x4
	.long	"??_R2?$numpunct@D@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2?$numpunct@D@std@@8"
	.globl	"??_R2?$numpunct@D@std@@8"      # @"??_R2?$numpunct@D@std@@8"
	.p2align	4
"??_R2?$numpunct@D@std@@8":
	.long	"??_R1A@?0A@EA@?$numpunct@D@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@facet@locale@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	"??_R17?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@?$numpunct@D@std@@8"
	.globl	"??_R1A@?0A@EA@?$numpunct@D@std@@8" # @"??_R1A@?0A@EA@?$numpunct@D@std@@8"
	.p2align	4
"??_R1A@?0A@EA@?$numpunct@D@std@@8":
	.long	"??_R0?AV?$numpunct@D@std@@@8"@IMGREL
	.long	3                               # 0x3
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3?$numpunct@D@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_C@_05LAPONLG@false?$AA@"
	.globl	"??_C@_05LAPONLG@false?$AA@"    # @"??_C@_05LAPONLG@false?$AA@"
"??_C@_05LAPONLG@false?$AA@":
	.asciz	"false"

	.section	.rdata,"dr",discard,"??_C@_04LOAJBDKD@true?$AA@"
	.globl	"??_C@_04LOAJBDKD@true?$AA@"    # @"??_C@_04LOAJBDKD@true?$AA@"
"??_C@_04LOAJBDKD@true?$AA@":
	.asciz	"true"

	.section	.rdata,"dr"
"?_Fake_alloc@std@@3U_Fake_allocator@1@B": # @"?_Fake_alloc@std@@3U_Fake_allocator@1@B"
	.zero	1

	.section	.rdata,"dr",discard,"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@"
	.globl	"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@" # @"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@"
"??_C@_0BA@JFNIOLAK@string?5too?5long?$AA@":
	.asciz	"string too long"

	.section	.data,"dw",discard,"??_R0?AVbad_array_new_length@std@@@8"
	.globl	"??_R0?AVbad_array_new_length@std@@@8" # @"??_R0?AVbad_array_new_length@std@@@8"
	.p2align	4
"??_R0?AVbad_array_new_length@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVbad_array_new_length@std@@"
	.zero	1

	.section	.xdata,"dr",discard,"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24"
	.globl	"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24" # @"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24"
	.p2align	4
"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24":
	.long	0                               # 0x0
	.long	"??_R0?AVbad_array_new_length@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	24                              # 0x18
	.long	"??0bad_array_new_length@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVbad_alloc@std@@@8"
	.globl	"??_R0?AVbad_alloc@std@@@8"     # @"??_R0?AVbad_alloc@std@@@8"
	.p2align	4
"??_R0?AVbad_alloc@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVbad_alloc@std@@"
	.zero	4

	.section	.xdata,"dr",discard,"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24"
	.globl	"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24" # @"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24"
	.p2align	4
"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24":
	.long	16                              # 0x10
	.long	"??_R0?AVbad_alloc@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	24                              # 0x18
	.long	"??0bad_alloc@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.xdata,"dr",discard,"_CTA3?AVbad_array_new_length@std@@"
	.globl	"_CTA3?AVbad_array_new_length@std@@" # @"_CTA3?AVbad_array_new_length@std@@"
	.p2align	3
"_CTA3?AVbad_array_new_length@std@@":
	.long	3                               # 0x3
	.long	"_CT??_R0?AVbad_array_new_length@std@@@8??0bad_array_new_length@std@@QEAA@AEBV01@@Z24"@IMGREL
	.long	"_CT??_R0?AVbad_alloc@std@@@8??0bad_alloc@std@@QEAA@AEBV01@@Z24"@IMGREL
	.long	"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24"@IMGREL

	.section	.xdata,"dr",discard,"_TI3?AVbad_array_new_length@std@@"
	.globl	"_TI3?AVbad_array_new_length@std@@" # @"_TI3?AVbad_array_new_length@std@@"
	.p2align	3
"_TI3?AVbad_array_new_length@std@@":
	.long	0                               # 0x0
	.long	"??1bad_array_new_length@std@@UEAA@XZ"@IMGREL
	.long	0                               # 0x0
	.long	"_CTA3?AVbad_array_new_length@std@@"@IMGREL

	.section	.rdata,"dr",discard,"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@"
	.globl	"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@" # @"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@"
"??_C@_0BF@KINCDENJ@bad?5array?5new?5length?$AA@":
	.asciz	"bad array new length"

	.section	.rdata,"dr",largest,"??_7bad_array_new_length@std@@6B@"
	.p2align	4                               # @8
.L__unnamed_9:
	.quad	"??_R4bad_array_new_length@std@@6B@"
	.quad	"??_Gbad_array_new_length@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4bad_array_new_length@std@@6B@"
	.globl	"??_R4bad_array_new_length@std@@6B@" # @"??_R4bad_array_new_length@std@@6B@"
	.p2align	4
"??_R4bad_array_new_length@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVbad_array_new_length@std@@@8"@IMGREL
	.long	"??_R3bad_array_new_length@std@@8"@IMGREL
	.long	"??_R4bad_array_new_length@std@@6B@"@IMGREL

	.section	.rdata,"dr",discard,"??_R3bad_array_new_length@std@@8"
	.globl	"??_R3bad_array_new_length@std@@8" # @"??_R3bad_array_new_length@std@@8"
	.p2align	3
"??_R3bad_array_new_length@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	3                               # 0x3
	.long	"??_R2bad_array_new_length@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2bad_array_new_length@std@@8"
	.globl	"??_R2bad_array_new_length@std@@8" # @"??_R2bad_array_new_length@std@@8"
	.p2align	2
"??_R2bad_array_new_length@std@@8":
	.long	"??_R1A@?0A@EA@bad_array_new_length@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@bad_alloc@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@bad_array_new_length@std@@8"
	.globl	"??_R1A@?0A@EA@bad_array_new_length@std@@8" # @"??_R1A@?0A@EA@bad_array_new_length@std@@8"
	.p2align	4
"??_R1A@?0A@EA@bad_array_new_length@std@@8":
	.long	"??_R0?AVbad_array_new_length@std@@@8"@IMGREL
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3bad_array_new_length@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@bad_alloc@std@@8"
	.globl	"??_R1A@?0A@EA@bad_alloc@std@@8" # @"??_R1A@?0A@EA@bad_alloc@std@@8"
	.p2align	4
"??_R1A@?0A@EA@bad_alloc@std@@8":
	.long	"??_R0?AVbad_alloc@std@@@8"@IMGREL
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3bad_alloc@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R3bad_alloc@std@@8"
	.globl	"??_R3bad_alloc@std@@8"         # @"??_R3bad_alloc@std@@8"
	.p2align	3
"??_R3bad_alloc@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	"??_R2bad_alloc@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2bad_alloc@std@@8"
	.globl	"??_R2bad_alloc@std@@8"         # @"??_R2bad_alloc@std@@8"
	.p2align	2
"??_R2bad_alloc@std@@8":
	.long	"??_R1A@?0A@EA@bad_alloc@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",largest,"??_7bad_alloc@std@@6B@"
	.p2align	4                               # @9
.L__unnamed_10:
	.quad	"??_R4bad_alloc@std@@6B@"
	.quad	"??_Gbad_alloc@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4bad_alloc@std@@6B@"
	.globl	"??_R4bad_alloc@std@@6B@"       # @"??_R4bad_alloc@std@@6B@"
	.p2align	4
"??_R4bad_alloc@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVbad_alloc@std@@@8"@IMGREL
	.long	"??_R3bad_alloc@std@@8"@IMGREL
	.long	"??_R4bad_alloc@std@@6B@"@IMGREL

	.section	.rdata,"dr",discard,"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"
	.globl	"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB" # @"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"
"?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB":
	.asciz	"0123456789-+Ee"

	.section	.rdata,"dr",discard,"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"
	.globl	"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB" # @"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"
	.p2align	4
"?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB":
	.asciz	"0123456789ABCDEFabcdef-+XxPp"

	.section	.rdata,"dr",discard,"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
	.globl	"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@" # @"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@"
"??_C@_0BF@PHHKMMFD@ios_base?3?3badbit?5set?$AA@":
	.asciz	"ios_base::badbit set"

	.section	.rdata,"dr",discard,"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
	.globl	"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@" # @"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@"
"??_C@_0BG@FMKFHCIL@ios_base?3?3failbit?5set?$AA@":
	.asciz	"ios_base::failbit set"

	.section	.rdata,"dr",discard,"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@"
	.globl	"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@" # @"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@"
"??_C@_0BF@OOHOMBOF@ios_base?3?3eofbit?5set?$AA@":
	.asciz	"ios_base::eofbit set"

	.section	.data,"dw",discard,"??_R0?AVfailure@ios_base@std@@@8"
	.globl	"??_R0?AVfailure@ios_base@std@@@8" # @"??_R0?AVfailure@ios_base@std@@@8"
	.p2align	4
"??_R0?AVfailure@ios_base@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVfailure@ios_base@std@@"
	.zero	5

	.section	.xdata,"dr",discard,"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40"
	.globl	"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40" # @"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40"
	.p2align	4
"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40":
	.long	0                               # 0x0
	.long	"??_R0?AVfailure@ios_base@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	40                              # 0x28
	.long	"??0failure@ios_base@std@@QEAA@AEBV012@@Z"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVsystem_error@std@@@8"
	.globl	"??_R0?AVsystem_error@std@@@8"  # @"??_R0?AVsystem_error@std@@@8"
	.p2align	4
"??_R0?AVsystem_error@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVsystem_error@std@@"
	.zero	1

	.section	.xdata,"dr",discard,"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40"
	.globl	"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40" # @"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40"
	.p2align	4
"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40":
	.long	0                               # 0x0
	.long	"??_R0?AVsystem_error@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	40                              # 0x28
	.long	"??0system_error@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV_System_error@std@@@8"
	.globl	"??_R0?AV_System_error@std@@@8" # @"??_R0?AV_System_error@std@@@8"
	.p2align	4
"??_R0?AV_System_error@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV_System_error@std@@"

	.section	.xdata,"dr",discard,"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40"
	.globl	"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40" # @"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40"
	.p2align	4
"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40":
	.long	0                               # 0x0
	.long	"??_R0?AV_System_error@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	40                              # 0x28
	.long	"??0_System_error@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVruntime_error@std@@@8"
	.globl	"??_R0?AVruntime_error@std@@@8" # @"??_R0?AVruntime_error@std@@@8"
	.p2align	4
"??_R0?AVruntime_error@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVruntime_error@std@@"

	.section	.xdata,"dr",discard,"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24"
	.globl	"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24" # @"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24"
	.p2align	4
"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24":
	.long	0                               # 0x0
	.long	"??_R0?AVruntime_error@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	24                              # 0x18
	.long	"??0runtime_error@std@@QEAA@AEBV01@@Z"@IMGREL

	.section	.xdata,"dr",discard,"_CTA5?AVfailure@ios_base@std@@"
	.globl	"_CTA5?AVfailure@ios_base@std@@" # @"_CTA5?AVfailure@ios_base@std@@"
	.p2align	4
"_CTA5?AVfailure@ios_base@std@@":
	.long	5                               # 0x5
	.long	"_CT??_R0?AVfailure@ios_base@std@@@8??0failure@ios_base@std@@QEAA@AEBV012@@Z40"@IMGREL
	.long	"_CT??_R0?AVsystem_error@std@@@8??0system_error@std@@QEAA@AEBV01@@Z40"@IMGREL
	.long	"_CT??_R0?AV_System_error@std@@@8??0_System_error@std@@QEAA@AEBV01@@Z40"@IMGREL
	.long	"_CT??_R0?AVruntime_error@std@@@8??0runtime_error@std@@QEAA@AEBV01@@Z24"@IMGREL
	.long	"_CT??_R0?AVexception@std@@@8??0exception@std@@QEAA@AEBV01@@Z24"@IMGREL

	.section	.xdata,"dr",discard,"_TI5?AVfailure@ios_base@std@@"
	.globl	"_TI5?AVfailure@ios_base@std@@" # @"_TI5?AVfailure@ios_base@std@@"
	.p2align	3
"_TI5?AVfailure@ios_base@std@@":
	.long	0                               # 0x0
	.long	"??1failure@ios_base@std@@UEAA@XZ"@IMGREL
	.long	0                               # 0x0
	.long	"_CTA5?AVfailure@ios_base@std@@"@IMGREL

	.section	.data,"dw",discard,"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A"
	.globl	"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A" # @"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A"
	.p2align	3
"?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A":
	.quad	"??_7_Iostream_error_category2@std@@6B@"
	.quad	5                               # 0x5

	.section	.rdata,"dr",largest,"??_7_Iostream_error_category2@std@@6B@"
	.p2align	4                               # @10
.L__unnamed_11:
	.quad	"??_R4_Iostream_error_category2@std@@6B@"
	.quad	"??_G_Iostream_error_category2@std@@UEAAPEAXI@Z"
	.quad	"?name@_Iostream_error_category2@std@@UEBAPEBDXZ"
	.quad	"?message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@H@Z"
	.quad	"?default_error_condition@error_category@std@@UEBA?AVerror_condition@2@H@Z"
	.quad	"?equivalent@error_category@std@@UEBA_NAEBVerror_code@2@H@Z"
	.quad	"?equivalent@error_category@std@@UEBA_NHAEBVerror_condition@2@@Z"

	.section	.bss,"bw",discard,"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"
	.globl	"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA" # @"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"
	.p2align	2
"?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA":
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R4_Iostream_error_category2@std@@6B@"
	.globl	"??_R4_Iostream_error_category2@std@@6B@" # @"??_R4_Iostream_error_category2@std@@6B@"
	.p2align	4
"??_R4_Iostream_error_category2@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV_Iostream_error_category2@std@@@8"@IMGREL
	.long	"??_R3_Iostream_error_category2@std@@8"@IMGREL
	.long	"??_R4_Iostream_error_category2@std@@6B@"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV_Iostream_error_category2@std@@@8"
	.globl	"??_R0?AV_Iostream_error_category2@std@@@8" # @"??_R0?AV_Iostream_error_category2@std@@@8"
	.p2align	4
"??_R0?AV_Iostream_error_category2@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV_Iostream_error_category2@std@@"
	.zero	4

	.section	.rdata,"dr",discard,"??_R3_Iostream_error_category2@std@@8"
	.globl	"??_R3_Iostream_error_category2@std@@8" # @"??_R3_Iostream_error_category2@std@@8"
	.p2align	3
"??_R3_Iostream_error_category2@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	"??_R2_Iostream_error_category2@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2_Iostream_error_category2@std@@8"
	.globl	"??_R2_Iostream_error_category2@std@@8" # @"??_R2_Iostream_error_category2@std@@8"
	.p2align	2
"??_R2_Iostream_error_category2@std@@8":
	.long	"??_R1A@?0A@EA@_Iostream_error_category2@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@error_category@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@_Iostream_error_category2@std@@8"
	.globl	"??_R1A@?0A@EA@_Iostream_error_category2@std@@8" # @"??_R1A@?0A@EA@_Iostream_error_category2@std@@8"
	.p2align	4
"??_R1A@?0A@EA@_Iostream_error_category2@std@@8":
	.long	"??_R0?AV_Iostream_error_category2@std@@@8"@IMGREL
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3_Iostream_error_category2@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@error_category@std@@8"
	.globl	"??_R1A@?0A@EA@error_category@std@@8" # @"??_R1A@?0A@EA@error_category@std@@8"
	.p2align	4
"??_R1A@?0A@EA@error_category@std@@8":
	.long	"??_R0?AVerror_category@std@@@8"@IMGREL
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3error_category@std@@8"@IMGREL

	.section	.data,"dw",discard,"??_R0?AVerror_category@std@@@8"
	.globl	"??_R0?AVerror_category@std@@@8" # @"??_R0?AVerror_category@std@@@8"
	.p2align	4
"??_R0?AVerror_category@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AVerror_category@std@@"
	.zero	7

	.section	.rdata,"dr",discard,"??_R3error_category@std@@8"
	.globl	"??_R3error_category@std@@8"    # @"??_R3error_category@std@@8"
	.p2align	3
"??_R3error_category@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	"??_R2error_category@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2error_category@std@@8"
	.globl	"??_R2error_category@std@@8"    # @"??_R2error_category@std@@8"
	.p2align	2
"??_R2error_category@std@@8":
	.long	"??_R1A@?0A@EA@error_category@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_C@_08LLGCOLLL@iostream?$AA@"
	.globl	"??_C@_08LLGCOLLL@iostream?$AA@" # @"??_C@_08LLGCOLLL@iostream?$AA@"
"??_C@_08LLGCOLLL@iostream?$AA@":
	.asciz	"iostream"

	.section	.rdata,"dr",discard,"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB"
	.globl	"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB" # @"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB"
	.p2align	4
"?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB":
	.asciz	"iostream stream error"

	.section	.rdata,"dr",largest,"??_7failure@ios_base@std@@6B@"
	.p2align	4                               # @11
.L__unnamed_12:
	.quad	"??_R4failure@ios_base@std@@6B@"
	.quad	"??_Gfailure@ios_base@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4failure@ios_base@std@@6B@"
	.globl	"??_R4failure@ios_base@std@@6B@" # @"??_R4failure@ios_base@std@@6B@"
	.p2align	4
"??_R4failure@ios_base@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVfailure@ios_base@std@@@8"@IMGREL
	.long	"??_R3failure@ios_base@std@@8"@IMGREL
	.long	"??_R4failure@ios_base@std@@6B@"@IMGREL

	.section	.rdata,"dr",discard,"??_R3failure@ios_base@std@@8"
	.globl	"??_R3failure@ios_base@std@@8"  # @"??_R3failure@ios_base@std@@8"
	.p2align	3
"??_R3failure@ios_base@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	5                               # 0x5
	.long	"??_R2failure@ios_base@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2failure@ios_base@std@@8"
	.globl	"??_R2failure@ios_base@std@@8"  # @"??_R2failure@ios_base@std@@8"
	.p2align	4
"??_R2failure@ios_base@std@@8":
	.long	"??_R1A@?0A@EA@failure@ios_base@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@system_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_System_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@runtime_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@failure@ios_base@std@@8"
	.globl	"??_R1A@?0A@EA@failure@ios_base@std@@8" # @"??_R1A@?0A@EA@failure@ios_base@std@@8"
	.p2align	4
"??_R1A@?0A@EA@failure@ios_base@std@@8":
	.long	"??_R0?AVfailure@ios_base@std@@@8"@IMGREL
	.long	4                               # 0x4
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3failure@ios_base@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@system_error@std@@8"
	.globl	"??_R1A@?0A@EA@system_error@std@@8" # @"??_R1A@?0A@EA@system_error@std@@8"
	.p2align	4
"??_R1A@?0A@EA@system_error@std@@8":
	.long	"??_R0?AVsystem_error@std@@@8"@IMGREL
	.long	3                               # 0x3
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3system_error@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R3system_error@std@@8"
	.globl	"??_R3system_error@std@@8"      # @"??_R3system_error@std@@8"
	.p2align	3
"??_R3system_error@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	4                               # 0x4
	.long	"??_R2system_error@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2system_error@std@@8"
	.globl	"??_R2system_error@std@@8"      # @"??_R2system_error@std@@8"
	.p2align	4
"??_R2system_error@std@@8":
	.long	"??_R1A@?0A@EA@system_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_System_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@runtime_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@_System_error@std@@8"
	.globl	"??_R1A@?0A@EA@_System_error@std@@8" # @"??_R1A@?0A@EA@_System_error@std@@8"
	.p2align	4
"??_R1A@?0A@EA@_System_error@std@@8":
	.long	"??_R0?AV_System_error@std@@@8"@IMGREL
	.long	2                               # 0x2
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3_System_error@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R3_System_error@std@@8"
	.globl	"??_R3_System_error@std@@8"     # @"??_R3_System_error@std@@8"
	.p2align	3
"??_R3_System_error@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	3                               # 0x3
	.long	"??_R2_System_error@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2_System_error@std@@8"
	.globl	"??_R2_System_error@std@@8"     # @"??_R2_System_error@std@@8"
	.p2align	2
"??_R2_System_error@std@@8":
	.long	"??_R1A@?0A@EA@_System_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@runtime_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@runtime_error@std@@8"
	.globl	"??_R1A@?0A@EA@runtime_error@std@@8" # @"??_R1A@?0A@EA@runtime_error@std@@8"
	.p2align	4
"??_R1A@?0A@EA@runtime_error@std@@8":
	.long	"??_R0?AVruntime_error@std@@@8"@IMGREL
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3runtime_error@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R3runtime_error@std@@8"
	.globl	"??_R3runtime_error@std@@8"     # @"??_R3runtime_error@std@@8"
	.p2align	3
"??_R3runtime_error@std@@8":
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	2                               # 0x2
	.long	"??_R2runtime_error@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2runtime_error@std@@8"
	.globl	"??_R2runtime_error@std@@8"     # @"??_R2runtime_error@std@@8"
	.p2align	2
"??_R2runtime_error@std@@8":
	.long	"??_R1A@?0A@EA@runtime_error@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@exception@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",largest,"??_7system_error@std@@6B@"
	.p2align	4                               # @12
.L__unnamed_13:
	.quad	"??_R4system_error@std@@6B@"
	.quad	"??_Gsystem_error@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4system_error@std@@6B@"
	.globl	"??_R4system_error@std@@6B@"    # @"??_R4system_error@std@@6B@"
	.p2align	4
"??_R4system_error@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVsystem_error@std@@@8"@IMGREL
	.long	"??_R3system_error@std@@8"@IMGREL
	.long	"??_R4system_error@std@@6B@"@IMGREL

	.section	.rdata,"dr",largest,"??_7_System_error@std@@6B@"
	.p2align	4                               # @13
.L__unnamed_14:
	.quad	"??_R4_System_error@std@@6B@"
	.quad	"??_G_System_error@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4_System_error@std@@6B@"
	.globl	"??_R4_System_error@std@@6B@"   # @"??_R4_System_error@std@@6B@"
	.p2align	4
"??_R4_System_error@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV_System_error@std@@@8"@IMGREL
	.long	"??_R3_System_error@std@@8"@IMGREL
	.long	"??_R4_System_error@std@@6B@"@IMGREL

	.section	.rdata,"dr",discard,"??_C@_02LMMGGCAJ@?3?5?$AA@"
	.globl	"??_C@_02LMMGGCAJ@?3?5?$AA@"    # @"??_C@_02LMMGGCAJ@?3?5?$AA@"
"??_C@_02LMMGGCAJ@?3?5?$AA@":
	.asciz	": "

	.section	.rdata,"dr",largest,"??_7runtime_error@std@@6B@"
	.p2align	4                               # @14
.L__unnamed_15:
	.quad	"??_R4runtime_error@std@@6B@"
	.quad	"??_Gruntime_error@std@@UEAAPEAXI@Z"
	.quad	"?what@exception@std@@UEBAPEBDXZ"

	.section	.rdata,"dr",discard,"??_R4runtime_error@std@@6B@"
	.globl	"??_R4runtime_error@std@@6B@"   # @"??_R4runtime_error@std@@6B@"
	.p2align	4
"??_R4runtime_error@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AVruntime_error@std@@@8"@IMGREL
	.long	"??_R3runtime_error@std@@8"@IMGREL
	.long	"??_R4runtime_error@std@@6B@"@IMGREL

	.section	.bss,"bw",discard,"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"
	.globl	"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB" # @"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"
	.p2align	3
"?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB":
	.quad	0

	.section	.rdata,"dr",largest,"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.p2align	4                               # @15
.L__unnamed_16:
	.quad	"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.quad	"??_G?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEAAPEAXI@Z"
	.quad	"?_Incref@facet@locale@std@@UEAAXXZ"
	.quad	"?_Decref@facet@locale@std@@UEAAPEAV_Facet_base@3@XZ"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBX@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DO@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DN@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_K@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_J@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DK@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
	.quad	"?do_put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@MEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@D_N@Z"

	.section	.rdata,"dr",discard,"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.globl	"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@" # @"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.p2align	4
"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@":
	.long	1                               # 0x1
	.long	0                               # 0x0
	.long	0                               # 0x0
	.long	"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"@IMGREL
	.long	"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL
	.long	"??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"@IMGREL

	.section	.data,"dw",discard,"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"
	.globl	"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8" # @"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"
	.p2align	4
"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8":
	.quad	"??_7type_info@@6B@"
	.quad	0
	.asciz	".?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@"
	.zero	6

	.section	.rdata,"dr",discard,"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.globl	"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" # @"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.p2align	3
"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8":
	.long	0                               # 0x0
	.long	1                               # 0x1
	.long	4                               # 0x4
	.long	"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.globl	"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" # @"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.p2align	4
"??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8":
	.long	"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@facet@locale@std@@8"@IMGREL
	.long	"??_R1A@?0A@EA@_Facet_base@std@@8"@IMGREL
	.long	"??_R17?0A@EA@_Crt_new_delete@std@@8"@IMGREL
	.long	0                               # 0x0

	.section	.rdata,"dr",discard,"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.globl	"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8" # @"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.p2align	4
"??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8":
	.long	"??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"@IMGREL
	.long	3                               # 0x3
	.long	0                               # 0x0
	.long	4294967295                      # 0xffffffff
	.long	0                               # 0x0
	.long	64                              # 0x40
	.long	"??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"@IMGREL

	.section	.rdata,"dr",discard,"??_C@_02BBAHNLBA@?$CFp?$AA@"
	.globl	"??_C@_02BBAHNLBA@?$CFp?$AA@"   # @"??_C@_02BBAHNLBA@?$CFp?$AA@"
"??_C@_02BBAHNLBA@?$CFp?$AA@":
	.asciz	"%p"

	.section	.rdata,"dr",discard,"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@"
	.globl	"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@" # @"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@"
"??_C@_0BI@CFPLBAOH@invalid?5string?5position?$AA@":
	.asciz	"invalid string position"

	.section	.bss,"bw",discard,"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA"
	.globl	"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA" # @"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA"
	.p2align	3
"?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA":
	.quad	0                               # 0x0

	.section	.rdata,"dr",discard,"??_C@_02MDKMJEGG@eE?$AA@"
	.globl	"??_C@_02MDKMJEGG@eE?$AA@"      # @"??_C@_02MDKMJEGG@eE?$AA@"
"??_C@_02MDKMJEGG@eE?$AA@":
	.asciz	"eE"

	.section	.rdata,"dr",discard,"??_C@_02OOPEBDOJ@pP?$AA@"
	.globl	"??_C@_02OOPEBDOJ@pP?$AA@"      # @"??_C@_02OOPEBDOJ@pP?$AA@"
"??_C@_02OOPEBDOJ@pP?$AA@":
	.asciz	"pP"

	.section	.rdata,"dr"
".L__const.?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z._Dp": # @"__const.?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z._Dp"
	.asciz	"."

	.section	.rdata,"dr",discard,"??_C@_02CLHGNPPK@Lu?$AA@"
	.globl	"??_C@_02CLHGNPPK@Lu?$AA@"      # @"??_C@_02CLHGNPPK@Lu?$AA@"
"??_C@_02CLHGNPPK@Lu?$AA@":
	.asciz	"Lu"

	.section	.rdata,"dr",discard,"??_C@_02HIKPPMOK@Ld?$AA@"
	.globl	"??_C@_02HIKPPMOK@Ld?$AA@"      # @"??_C@_02HIKPPMOK@Ld?$AA@"
"??_C@_02HIKPPMOK@Ld?$AA@":
	.asciz	"Ld"

	.section	.rdata,"dr",discard,"??_C@_02BDDLJJBK@lu?$AA@"
	.globl	"??_C@_02BDDLJJBK@lu?$AA@"      # @"??_C@_02BDDLJJBK@lu?$AA@"
"??_C@_02BDDLJJBK@lu?$AA@":
	.asciz	"lu"

	.section	.rdata,"dr",discard,"??_C@_02EAOCLKAK@ld?$AA@"
	.globl	"??_C@_02EAOCLKAK@ld?$AA@"      # @"??_C@_02EAOCLKAK@ld?$AA@"
"??_C@_02EAOCLKAK@ld?$AA@":
	.asciz	"ld"

	.section	.CRT$XCU,"dr",associative,"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.p2align	3
	.quad	"??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
	.section	.CRT$XCU,"dr",associative,"?id@?$numpunct@D@std@@2V0locale@2@A"
	.p2align	3
	.quad	"??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ"
	.section	.CRT$XCU,"dr",associative,"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.p2align	3
	.quad	"??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
	.section	.drectve,"yn"
	.ascii	" /FAILIFMISMATCH:\"_MSC_VER=1900\""
	.ascii	" /FAILIFMISMATCH:\"_ITERATOR_DEBUG_LEVEL=0\""
	.ascii	" /FAILIFMISMATCH:\"RuntimeLibrary=MT_StaticRelease\""
	.ascii	" /DEFAULTLIB:libcpmt.lib"
	.ascii	" /FAILIFMISMATCH:\"_CRT_STDIO_ISO_WIDE_SPECIFIERS=0\""
	.ascii	" /FAILIFMISMATCH:\"annotate_string=0\""
	.ascii	" /INCLUDE:\"?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A\""
	.ascii	" /INCLUDE:\"?id@?$numpunct@D@std@@2V0locale@2@A\""
	.ascii	" /INCLUDE:\"?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A\""
	.globl	"??_7?$ctype@D@std@@6B@"
.set "??_7?$ctype@D@std@@6B@", .L__unnamed_1+8
	.globl	"??_7ctype_base@std@@6B@"
.set "??_7ctype_base@std@@6B@", .L__unnamed_2+8
	.globl	"??_7facet@locale@std@@6B@"
.set "??_7facet@locale@std@@6B@", .L__unnamed_3+8
	.globl	"??_7_Facet_base@std@@6B@"
.set "??_7_Facet_base@std@@6B@", .L__unnamed_4+8
	.globl	"??_7bad_cast@std@@6B@"
.set "??_7bad_cast@std@@6B@", .L__unnamed_5+8
	.globl	"??_7exception@std@@6B@"
.set "??_7exception@std@@6B@", .L__unnamed_6+8
	.globl	"??_7?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
.set "??_7?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@", .L__unnamed_7+8
	.globl	"??_7?$numpunct@D@std@@6B@"
.set "??_7?$numpunct@D@std@@6B@", .L__unnamed_8+8
	.globl	"??_7bad_array_new_length@std@@6B@"
.set "??_7bad_array_new_length@std@@6B@", .L__unnamed_9+8
	.globl	"??_7bad_alloc@std@@6B@"
.set "??_7bad_alloc@std@@6B@", .L__unnamed_10+8
	.globl	"??_7_Iostream_error_category2@std@@6B@"
.set "??_7_Iostream_error_category2@std@@6B@", .L__unnamed_11+8
	.globl	"??_7failure@ios_base@std@@6B@"
.set "??_7failure@ios_base@std@@6B@", .L__unnamed_12+8
	.globl	"??_7system_error@std@@6B@"
.set "??_7system_error@std@@6B@", .L__unnamed_13+8
	.globl	"??_7_System_error@std@@6B@"
.set "??_7_System_error@std@@6B@", .L__unnamed_14+8
	.globl	"??_7runtime_error@std@@6B@"
.set "??_7runtime_error@std@@6B@", .L__unnamed_15+8
	.globl	"??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
.set "??_7?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@", .L__unnamed_16+8
	.addrsig
	.addrsig_sym "?f@@YAXHH@Z"
	.addrsig_sym "?f_cpp@@YAHHH@Z"
	.addrsig_sym "??5?$basic_istream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@AEAH@Z"
	.addrsig_sym "??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@H@Z"
	.addrsig_sym "??6?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV01@P6AAEAV01@AEAV01@@Z@Z"
	.addrsig_sym "??$endl@DU?$char_traits@D@std@@@std@@YAAEAV?$basic_ostream@DU?$char_traits@D@std@@@0@AEAV10@@Z"
	.addrsig_sym "??__E?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
	.addrsig_sym "??__E?id@?$numpunct@D@std@@2V0locale@2@A@@YAXXZ"
	.addrsig_sym "??__E?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A@@YAXXZ"
	.addrsig_sym "??$_Common_extract_with_num_get@J@?$basic_istream@DU?$char_traits@D@std@@@std@@AEAAAEAV01@AEAJ@Z"
	.addrsig_sym "??Bsentry@?$basic_istream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	.addrsig_sym __CxxFrameHandler3
	.addrsig_sym "??$use_facet@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.addrsig_sym "?getloc@ios_base@std@@QEBA?AVlocale@2@XZ"
	.addrsig_sym "?get@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@V32@0AEAVios_base@2@AEAHAEAJ@Z"
	.addrsig_sym "?setstate@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	.addrsig_sym "?_Ipfx@?$basic_istream@DU?$char_traits@D@std@@@std@@QEAA_N_N@Z"
	.addrsig_sym "?rdbuf@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_streambuf@DU?$char_traits@D@std@@@2@XZ"
	.addrsig_sym "?good@ios_base@std@@QEBA_NXZ"
	.addrsig_sym "?tie@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBAPEAV?$basic_ostream@DU?$char_traits@D@std@@@2@XZ"
	.addrsig_sym "?flush@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@XZ"
	.addrsig_sym "?flags@ios_base@std@@QEBAHXZ"
	.addrsig_sym "??$use_facet@V?$ctype@D@std@@@std@@YAAEBV?$ctype@D@0@AEBVlocale@0@@Z"
	.addrsig_sym "?sgetc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.addrsig_sym "?eq_int_type@?$_Narrow_char_traits@DH@std@@SA_NAEBH0@Z"
	.addrsig_sym "?eof@?$_Narrow_char_traits@DH@std@@SAHXZ"
	.addrsig_sym "?is@?$ctype@D@std@@QEBA_NFD@Z"
	.addrsig_sym "?to_char_type@?$_Narrow_char_traits@DH@std@@SADAEBH@Z"
	.addrsig_sym "?snextc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.addrsig_sym "?rdstate@ios_base@std@@QEBAHXZ"
	.addrsig_sym "??Bsentry@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	.addrsig_sym "?pubsync@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.addrsig_sym __std_terminate
	.addrsig_sym "?uncaught_exception@std@@YA_NXZ"
	.addrsig_sym "?_Osfx@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAXXZ"
	.addrsig_sym "??Bid@locale@std@@QEAA_KXZ"
	.addrsig_sym "?_Getfacet@locale@std@@QEBAPEBVfacet@12@_K@Z"
	.addrsig_sym "?_Getcat@?$ctype@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.addrsig_sym "?_Throw_bad_cast@std@@YAXXZ"
	.addrsig_sym "?_Facet_Register@std@@YAXPEAV_Facet_base@1@@Z"
	.addrsig_sym "?release@?$unique_ptr@V_Facet_base@std@@U?$default_delete@V_Facet_base@std@@@2@@std@@QEAAPEAV_Facet_base@2@XZ"
	.addrsig_sym "?_Getgloballocale@locale@std@@CAPEAV_Locimp@12@XZ"
	.addrsig_sym "??2@YAPEAX_K@Z"
	.addrsig_sym "?_C_str@locale@std@@QEBAPEBDXZ"
	.addrsig_sym "??3@YAXPEAX@Z"
	.addrsig_sym "?c_str@?$_Yarn@D@std@@QEBAPEBDXZ"
	.addrsig_sym "?_Locinfo_ctor@_Locinfo@std@@SAXPEAV12@PEBD@Z"
	.addrsig_sym "?_Xruntime_error@std@@YAXPEBD@Z"
	.addrsig_sym "?_Tidy@?$_Yarn@D@std@@AEAAXXZ"
	.addrsig_sym free
	.addrsig_sym "?_Tidy@?$_Yarn@_W@std@@AEAAXXZ"
	.addrsig_sym "?_Init@?$ctype@D@std@@IEAAXAEBV_Locinfo@2@@Z"
	.addrsig_sym "?_Getctype@_Locinfo@std@@QEBA?AU_Ctypevec@@XZ"
	.addrsig_sym _Getctype
	.addrsig_sym "?_Tidy@?$ctype@D@std@@IEAAXXZ"
	.addrsig_sym "??_V@YAXPEAX@Z"
	.addrsig_sym "??$_Adl_verify_range@PEADPEBD@std@@YAXAEBQEADAEBQEBD@Z"
	.addrsig_sym _Tolower
	.addrsig_sym _Toupper
	.addrsig_sym "??$_Adl_verify_range@PEBDPEBD@std@@YAXAEBQEBD0@Z"
	.addrsig_sym "?_Locinfo_dtor@_Locinfo@std@@SAXPEAV12@@Z"
	.addrsig_sym _CxxThrowException
	.addrsig_sym __std_exception_destroy
	.addrsig_sym __std_exception_copy
	.addrsig_sym "??$exchange@PEAV_Facet_base@std@@$$T@std@@YAPEAV_Facet_base@0@AEAPEAV10@$$QEA$$T@Z"
	.addrsig_sym "?_Get_first@?$_Compressed_pair@U?$default_delete@V_Facet_base@std@@@std@@PEAV_Facet_base@2@$00@std@@QEAAAEAU?$default_delete@V_Facet_base@std@@@2@XZ"
	.addrsig_sym "??R?$default_delete@V_Facet_base@std@@@std@@QEBAXPEAV_Facet_base@1@@Z"
	.addrsig_sym "?_Gnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	.addrsig_sym "?to_int_type@?$_Narrow_char_traits@DH@std@@SAHAEBD@Z"
	.addrsig_sym "?gptr@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBAPEADXZ"
	.addrsig_sym "?_Gnpreinc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	.addrsig_sym "?sbumpc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHXZ"
	.addrsig_sym "?_Gninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	.addrsig_sym "?_Getcat@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.addrsig_sym "?_Init@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
	.addrsig_sym "??$_Adl_verify_range@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@V12@@std@@YAXAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	.addrsig_sym "?_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1HAEBVlocale@2@@Z"
	.addrsig_sym _Stoullx
	.addrsig_sym "??$?8DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	.addrsig_sym "??$use_facet@V?$numpunct@D@std@@@std@@YAAEBV?$numpunct@D@0@AEBVlocale@0@@Z"
	.addrsig_sym "?grouping@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.addrsig_sym "?empty@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_NXZ"
	.addrsig_sym "?thousands_sep@?$numpunct@D@std@@QEBADXZ"
	.addrsig_sym "?widen@?$ctype@D@std@@QEBAPEBDPEBD0PEAD@Z"
	.addrsig_sym "??$end@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
	.addrsig_sym "??$begin@$$CBD$0BL@@std@@YAPEBDAEAY0BL@$$CBD@Z"
	.addrsig_sym "??$?9DU?$char_traits@D@std@@@std@@YA_NAEBV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0@Z"
	.addrsig_sym "??D?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBADXZ"
	.addrsig_sym "??E?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	.addrsig_sym "??$_Find_elem@D$0BL@@std@@YA_KAEAY0BL@$$CBDD@Z"
	.addrsig_sym "??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAD_K@Z"
	.addrsig_sym "?push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z"
	.addrsig_sym "??A?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAAEBD_K@Z"
	.addrsig_sym "?_Getcat@?$numpunct@D@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.addrsig_sym "?_Init@?$numpunct@D@std@@IEAAXAEBV_Locinfo@2@_N@Z"
	.addrsig_sym "?_Getlconv@_Locinfo@std@@QEBAPEBUlconv@@XZ"
	.addrsig_sym "?_Getcvt@_Locinfo@std@@QEBA?AU_Cvtvec@@XZ"
	.addrsig_sym "??$_Maklocstr@D@std@@YAPEADPEBDPEADAEBU_Cvtvec@@@Z"
	.addrsig_sym "?_Getfalse@_Locinfo@std@@QEBAPEBDXZ"
	.addrsig_sym "?_Gettrue@_Locinfo@std@@QEBAPEBDXZ"
	.addrsig_sym "??$_Maklocchr@D@std@@YADDPEADAEBU_Cvtvec@@@Z"
	.addrsig_sym "??$_Getvals@D@?$numpunct@D@std@@IEAAXDPEBUlconv@@U_Cvtvec@@@Z"
	.addrsig_sym localeconv
	.addrsig_sym _Getcvt
	.addrsig_sym strlen
	.addrsig_sym calloc
	.addrsig_sym "?_Xbad_alloc@std@@YAXXZ"
	.addrsig_sym "?_Tidy@?$numpunct@D@std@@AEAAXXZ"
	.addrsig_sym "??$_Construct@$00PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
	.addrsig_sym "??$_Convert_size@_K_K@std@@YA_K_K@Z"
	.addrsig_sym "?length@?$_Narrow_char_traits@DH@std@@SA_KQEBD@Z"
	.addrsig_sym "?max_size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	.addrsig_sym "?_Xlen_string@std@@YAXXZ"
	.addrsig_sym "?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV?$allocator@D@2@XZ"
	.addrsig_sym "?move@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	.addrsig_sym "?assign@?$_Narrow_char_traits@DH@std@@SAXAEADAEBD@Z"
	.addrsig_sym "?_Release@_Fake_proxy_ptr_impl@std@@QEAAXXZ"
	.addrsig_sym "?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBA_K_K@Z"
	.addrsig_sym "?allocate@?$allocator@D@std@@QEAAPEAD_K@Z"
	.addrsig_sym "??$_Construct_in_place@PEADAEBQEAD@std@@YAXAEAPEADAEBQEAD@Z"
	.addrsig_sym "?copy@?$_Char_traits@DH@std@@SAPEADQEADQEBD_K@Z"
	.addrsig_sym "??$_Unfancy@D@std@@YAPEADPEAD@Z"
	.addrsig_sym "?max_size@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA_KAEBV?$allocator@D@2@@Z"
	.addrsig_sym "?_Getal@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEBAAEBV?$allocator@D@2@XZ"
	.addrsig_sym "??$max@_K@std@@YAAEB_KAEB_K0@Z"
	.addrsig_sym "??$min@_K@std@@YAAEB_KAEB_K0@Z"
	.addrsig_sym "?max@?$numeric_limits@_J@std@@SA_JXZ"
	.addrsig_sym "?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEBAAEBV?$allocator@D@2@XZ"
	.addrsig_sym "?_Xlength_error@std@@YAXPEBD@Z"
	.addrsig_sym "?_Get_first@?$_Compressed_pair@V?$allocator@D@std@@V?$_String_val@U?$_Simple_types@D@std@@@2@$00@std@@QEAAAEAV?$allocator@D@2@XZ"
	.addrsig_sym "?_Calculate_growth@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@CA_K_K00@Z"
	.addrsig_sym "??$_Allocate@$0BA@U_Default_allocate_traits@std@@$0A@@std@@YAPEAX_K@Z"
	.addrsig_sym "??$_Get_size_of_n@$00@std@@YA_K_K@Z"
	.addrsig_sym "??$_Allocate_manually_vector_aligned@U_Default_allocate_traits@std@@@std@@YAPEAX_K@Z"
	.addrsig_sym "?_Allocate@_Default_allocate_traits@std@@SAPEAX_K@Z"
	.addrsig_sym "?_Throw_bad_array_new_length@std@@YAXXZ"
	.addrsig_sym _invalid_parameter_noinfo_noreturn
	.addrsig_sym "??$_Voidify_iter@PEAPEAD@std@@YAPEAXPEAPEAD@Z"
	.addrsig_sym "?_Peek@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEBADXZ"
	.addrsig_sym "?_Inc@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@AEAAXXZ"
	.addrsig_sym "??$_Construct@$0A@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXD_K@Z"
	.addrsig_sym "?assign@?$_Narrow_char_traits@DH@std@@SAPEADQEAD_KD@Z"
	.addrsig_sym "??$_Find_unchecked@PEBDD@std@@YAPEBDPEBDQEBDAEBD@Z"
	.addrsig_sym "??$_Could_compare_equal_to_value_type@PEBDD@std@@YA_NAEBD@Z"
	.addrsig_sym "??$_To_address@PEBD@std@@YA?A?<auto>@@AEBQEBD@Z"
	.addrsig_sym "??$__std_find_trivial@$$CBDD@@YAPEBDPEBD0D@Z"
	.addrsig_sym __std_find_trivial_1
	.addrsig_sym "?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAPEADXZ"
	.addrsig_sym "?_Large_string_engaged@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBA_NXZ"
	.addrsig_sym "??$_Reallocate_grow_by@V<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@D@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??push_back@01@QEAAXD@Z@D@Z"
	.addrsig_sym "?_Orphan_all@_Container_base0@std@@QEAAXXZ"
	.addrsig_sym "??R<lambda_1>@?0??push_back@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAXD@Z@QEBA?A?<auto>@@QEADQEBD_KD@Z"
	.addrsig_sym "?deallocate@?$allocator@D@std@@QEAAXQEAD_K@Z"
	.addrsig_sym "??$_Deallocate@$0BA@$0A@@std@@YAXPEAX_K@Z"
	.addrsig_sym "?_Adjust_manually_vector_aligned@std@@YAXAEAPEAXAEA_K@Z"
	.addrsig_sym "??3@YAXPEAX_K@Z"
	.addrsig_sym "?_Myptr@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAPEBDXZ"
	.addrsig_sym "?_Tidy_deallocate@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	.addrsig_sym "??$_Destroy_in_place@PEAD@std@@YAXAEAPEAD@Z"
	.addrsig_sym "?_Activate_SSO_buffer@?$_String_val@U?$_Simple_types@D@std@@@std@@QEAAXXZ"
	.addrsig_sym "?equal@?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NAEBV12@@Z"
	.addrsig_sym "?_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.addrsig_sym "?_Stodx_v2@std@@YANPEBDPEAPEADHPEAH@Z"
	.addrsig_sym ldexp
	.addrsig_sym "?_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@2@1AEAVios_base@2@PEAH@Z"
	.addrsig_sym "??$end@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
	.addrsig_sym "??$begin@$$CBD$0P@@std@@YAPEBDAEAY0P@$$CBD@Z"
	.addrsig_sym "??$_Find_elem@D$0P@@std@@YA_KAEAY0P@$$CBDD@Z"
	.addrsig_sym "?decimal_point@?$numpunct@D@std@@QEBADXZ"
	.addrsig_sym "??$end@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
	.addrsig_sym "??$begin@$$CBD$0BN@@std@@YAPEBDAEAY0BN@$$CBD@Z"
	.addrsig_sym "??$_Find_elem@D$0BN@@std@@YA_KAEAY0BN@$$CBDD@Z"
	.addrsig_sym _errno
	.addrsig_sym strtod
	.addrsig_sym pow
	.addrsig_sym "?_Stofx_v2@std@@YAMPEBDPEAPEADHPEAH@Z"
	.addrsig_sym ldexpf
	.addrsig_sym strtof
	.addrsig_sym powf
	.addrsig_sym _Stollx
	.addrsig_sym _Stoulx
	.addrsig_sym _Stolx
	.addrsig_sym "?falsename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.addrsig_sym "??Y?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@AEBV01@@Z"
	.addrsig_sym "?truename@?$numpunct@D@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.addrsig_sym "??$_Getloctxt@V?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@D@std@@YAHAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@0@0_KPEBDW4_Case_sensitive@0@@Z"
	.addrsig_sym "?c_str@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBAPEBDXZ"
	.addrsig_sym "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@AEBV12@@Z"
	.addrsig_sym "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD_K@Z"
	.addrsig_sym "??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@QEBD_K@Z@PEBD_K@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@QEBD0@Z@PEBD_K@Z"
	.addrsig_sym "??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@QEBD_K@Z@QEBA?A?<auto>@@QEAD0101@Z"
	.addrsig_sym "?tolower@?$ctype@D@std@@QEBADD@Z"
	.addrsig_sym "?_Init@locale@std@@CAPEAV_Locimp@12@_N@Z"
	.addrsig_sym "?clear@?$basic_ios@DU?$char_traits@D@std@@@std@@QEAAXH_N@Z"
	.addrsig_sym "?clear@ios_base@std@@QEAAXH_N@Z"
	.addrsig_sym "?make_error_code@std@@YA?AVerror_code@1@W4io_errc@1@@Z"
	.addrsig_sym "?iostream_category@std@@YAAEBVerror_category@1@XZ"
	.addrsig_sym "??$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ"
	.addrsig_sym _Init_thread_header
	.addrsig_sym "??__F_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@YAXXZ"
	.addrsig_sym atexit
	.addrsig_sym _Init_thread_footer
	.addrsig_sym "?_Syserror_map@std@@YAPEBDH@Z"
	.addrsig_sym "??8error_category@std@@QEBA_NAEBV01@@Z"
	.addrsig_sym "?category@error_code@std@@QEBAAEBVerror_category@2@XZ"
	.addrsig_sym "?value@error_code@std@@QEBAHXZ"
	.addrsig_sym "??$_Bit_cast@_KT_Addr_storage@error_category@std@@$0A@@std@@YA_KAEBT_Addr_storage@error_category@0@@Z"
	.addrsig_sym "??8std@@YA_NAEBVerror_condition@0@0@Z"
	.addrsig_sym "?category@error_condition@std@@QEBAAEBVerror_category@2@XZ"
	.addrsig_sym "?value@error_condition@std@@QEBAHXZ"
	.addrsig_sym "?_Makestr@_System_error@std@@CA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@Verror_code@2@V32@@Z"
	.addrsig_sym "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@QEBD@Z"
	.addrsig_sym "?message@error_code@std@@QEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@2@XZ"
	.addrsig_sym "?_Alloc_proxy@_Container_base0@std@@QEAAXAEBU_Fake_allocator@2@@Z"
	.addrsig_sym "?_Take_contents@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEAV12@@Z"
	.addrsig_sym "?_Memcpy_val_from@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXAEBV12@@Z"
	.addrsig_sym "?_Tidy_init@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXXZ"
	.addrsig_sym "?select_on_container_copy_construction@?$_Default_allocator_traits@V?$allocator@D@std@@@std@@SA?AV?$allocator@D@2@AEBV32@@Z"
	.addrsig_sym "??$_Construct@$01PEBD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAXQEBD_K@Z"
	.addrsig_sym "??$use_facet@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@YAAEBV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@0@AEBVlocale@0@@Z"
	.addrsig_sym "?put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@QEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DJ@Z"
	.addrsig_sym "?fill@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADXZ"
	.addrsig_sym "?failed@?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEBA_NXZ"
	.addrsig_sym "?_Getcat@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@SA_KPEAPEBVfacet@locale@2@PEBV42@@Z"
	.addrsig_sym "?_Init@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@IEAAXAEBV_Locinfo@2@@Z"
	.addrsig_sym "?_Iput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEAD_K@Z"
	.addrsig_sym sprintf_s
	.addrsig_sym "?insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_K0D@Z"
	.addrsig_sym "?size@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEBA_KXZ"
	.addrsig_sym "?width@ios_base@std@@QEBA_JXZ"
	.addrsig_sym "?_Rep@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@D_K@Z"
	.addrsig_sym "?_Put@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@PEBD_K@Z"
	.addrsig_sym "?width@ios_base@std@@QEAA_J_J@Z"
	.addrsig_sym "?_Check_offset@?$_String_val@U?$_Simple_types@D@std@@@std@@QEBAX_K@Z"
	.addrsig_sym "??$_Reallocate_grow_by@V<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_K0D@Z@_K_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??insert@01@QEAAAEAV01@00D@Z@_K2D@Z"
	.addrsig_sym "?_Xran@?$_String_val@U?$_Simple_types@D@std@@@std@@SAXXZ"
	.addrsig_sym "?_Xout_of_range@std@@YAXPEBD@Z"
	.addrsig_sym "??R<lambda_1>@?0??insert@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_K0D@Z@QEBA?A?<auto>@@QEADQEBD000D@Z"
	.addrsig_sym "??D?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	.addrsig_sym "??4?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@D@Z"
	.addrsig_sym "??E?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@QEAAAEAV01@XZ"
	.addrsig_sym "?sputc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@QEAAHD@Z"
	.addrsig_sym "?_Pnavail@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEBA_JXZ"
	.addrsig_sym "?_Pninc@?$basic_streambuf@DU?$char_traits@D@std@@@std@@IEAAPEADXZ"
	.addrsig_sym _vsprintf_s_l
	.addrsig_sym __stdio_common_vsprintf_s
	.addrsig_sym __local_stdio_printf_options
	.addrsig_sym "?precision@ios_base@std@@QEBA_JXZ"
	.addrsig_sym "??$_Float_put_desired_precision@O@std@@YAH_JH@Z"
	.addrsig_sym frexpl
	.addrsig_sym abs
	.addrsig_sym "?resize@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAX_KD@Z"
	.addrsig_sym "?_Ffmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADDH@Z"
	.addrsig_sym "?_Fput@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBA?AV?$ostreambuf_iterator@DU?$char_traits@D@std@@@2@V32@AEAVios_base@2@DPEBD_K@Z"
	.addrsig_sym frexp
	.addrsig_sym "?_Eos@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAX_K@Z"
	.addrsig_sym "?append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@_KD@Z"
	.addrsig_sym "??$_Reallocate_grow_by@V<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV34@_KD@Z@_KD@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@AEAAAEAV01@_KV<lambda_1>@?0??append@01@QEAAAEAV01@0D@Z@_KD@Z"
	.addrsig_sym "??R<lambda_1>@?0??append@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV23@_KD@Z@QEBA?A?<auto>@@QEADQEBD00D@Z"
	.addrsig_sym strcspn
	.addrsig_sym "??$_Float_put_desired_precision@N@std@@YAH_JH@Z"
	.addrsig_sym "?_Ifmt@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAPEADPEADPEBDH@Z"
	.addrsig_sym "?assign@?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV12@$$QEAV12@@Z"
	.addrsig_sym "??4?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@QEAAAEAV01@$$QEAV01@@Z"
	.addrsig_sym "??$_Pocma@V?$allocator@D@std@@@std@@YAXAEAV?$allocator@D@0@0@Z"
	.addrsig_sym "?put@?$basic_ostream@DU?$char_traits@D@std@@@std@@QEAAAEAV12@D@Z"
	.addrsig_sym "?widen@?$basic_ios@DU?$char_traits@D@std@@@std@@QEBADD@Z"
	.addrsig_sym "?widen@?$ctype@D@std@@QEBADD@Z"
	.addrsig_sym "?cin@std@@3V?$basic_istream@DU?$char_traits@D@std@@@1@A"
	.addrsig_sym "?cout@std@@3V?$basic_ostream@DU?$char_traits@D@std@@@1@A"
	.addrsig_sym "?id@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.addrsig_sym "?id@?$numpunct@D@std@@2V0locale@2@A"
	.addrsig_sym "?id@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@2V0locale@2@A"
	.addrsig_sym "?_Psave@?$_Facetptr@V?$ctype@D@std@@@std@@2PEBVfacet@locale@2@EB"
	.addrsig_sym "?id@?$ctype@D@std@@2V0locale@2@A"
	.addrsig_sym "?_Id_cnt@id@locale@std@@0HA"
	.addrsig_sym "??_R4?$ctype@D@std@@6B@"
	.addrsig_sym "??_7type_info@@6B@"
	.addrsig_sym "??_R0?AV?$ctype@D@std@@@8"
	.addrsig_sym __ImageBase
	.addrsig_sym "??_R3?$ctype@D@std@@8"
	.addrsig_sym "??_R2?$ctype@D@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@?$ctype@D@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@ctype_base@std@@8"
	.addrsig_sym "??_R0?AUctype_base@std@@@8"
	.addrsig_sym "??_R3ctype_base@std@@8"
	.addrsig_sym "??_R2ctype_base@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@facet@locale@std@@8"
	.addrsig_sym "??_R0?AVfacet@locale@std@@@8"
	.addrsig_sym "??_R3facet@locale@std@@8"
	.addrsig_sym "??_R2facet@locale@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@_Facet_base@std@@8"
	.addrsig_sym "??_R0?AV_Facet_base@std@@@8"
	.addrsig_sym "??_R3_Facet_base@std@@8"
	.addrsig_sym "??_R2_Facet_base@std@@8"
	.addrsig_sym "??_R17?0A@EA@_Crt_new_delete@std@@8"
	.addrsig_sym "??_R0?AU_Crt_new_delete@std@@@8"
	.addrsig_sym "??_R3_Crt_new_delete@std@@8"
	.addrsig_sym "??_R2_Crt_new_delete@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@_Crt_new_delete@std@@8"
	.addrsig_sym "??_R4ctype_base@std@@6B@"
	.addrsig_sym "??_R4facet@locale@std@@6B@"
	.addrsig_sym "??_R4_Facet_base@std@@6B@"
	.addrsig_sym "??_R0?AVbad_cast@std@@@8"
	.addrsig_sym "??_R0?AVexception@std@@@8"
	.addrsig_sym "??_R4bad_cast@std@@6B@"
	.addrsig_sym "??_R3bad_cast@std@@8"
	.addrsig_sym "??_R2bad_cast@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@bad_cast@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@exception@std@@8"
	.addrsig_sym "??_R3exception@std@@8"
	.addrsig_sym "??_R2exception@std@@8"
	.addrsig_sym "??_R4exception@std@@6B@"
	.addrsig_sym "?_Psave@?$_Facetptr@V?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"
	.addrsig_sym "??_R4?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.addrsig_sym "??_R0?AV?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"
	.addrsig_sym "??_R3?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.addrsig_sym "??_R2?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.addrsig_sym "?_Src@?1??_Getifld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1HAEBVlocale@3@@Z@4QBDB"
	.addrsig_sym "?_Psave@?$_Facetptr@V?$numpunct@D@std@@@std@@2PEBVfacet@locale@2@EB"
	.addrsig_sym "??_R4?$numpunct@D@std@@6B@"
	.addrsig_sym "??_R0?AV?$numpunct@D@std@@@8"
	.addrsig_sym "??_R3?$numpunct@D@std@@8"
	.addrsig_sym "??_R2?$numpunct@D@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@?$numpunct@D@std@@8"
	.addrsig_sym "?_Fake_alloc@std@@3U_Fake_allocator@1@B"
	.addrsig_sym "??_R0?AVbad_array_new_length@std@@@8"
	.addrsig_sym "??_R0?AVbad_alloc@std@@@8"
	.addrsig_sym "??_R4bad_array_new_length@std@@6B@"
	.addrsig_sym "??_R3bad_array_new_length@std@@8"
	.addrsig_sym "??_R2bad_array_new_length@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@bad_array_new_length@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@bad_alloc@std@@8"
	.addrsig_sym "??_R3bad_alloc@std@@8"
	.addrsig_sym "??_R2bad_alloc@std@@8"
	.addrsig_sym "??_R4bad_alloc@std@@6B@"
	.addrsig_sym "?_Src@?1??_Getffld@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"
	.addrsig_sym "?_Src@?1??_Getffldx@?$num_get@DV?$istreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@AEBAHPEADAEAV?$istreambuf_iterator@DU?$char_traits@D@std@@@3@1AEAVios_base@3@PEAH@Z@4QBDB"
	.addrsig_sym "??_R0?AVfailure@ios_base@std@@@8"
	.addrsig_sym "??_R0?AVsystem_error@std@@@8"
	.addrsig_sym "??_R0?AV_System_error@std@@@8"
	.addrsig_sym "??_R0?AVruntime_error@std@@@8"
	.addrsig_sym "?_Static@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@1@XZ@4V21@A"
	.addrsig_sym "?$TSS0@?1???$_Immortalize_memcpy_image@V_Iostream_error_category2@std@@@std@@YAAEBV_Iostream_error_category2@0@XZ@4HA"
	.addrsig_sym "??_R4_Iostream_error_category2@std@@6B@"
	.addrsig_sym "??_R0?AV_Iostream_error_category2@std@@@8"
	.addrsig_sym "??_R3_Iostream_error_category2@std@@8"
	.addrsig_sym "??_R2_Iostream_error_category2@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@_Iostream_error_category2@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@error_category@std@@8"
	.addrsig_sym "??_R0?AVerror_category@std@@@8"
	.addrsig_sym "??_R3error_category@std@@8"
	.addrsig_sym "??_R2error_category@std@@8"
	.addrsig_sym "?_Iostream_error@?4??message@_Iostream_error_category2@std@@UEBA?AV?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@3@H@Z@4QBDB"
	.addrsig_sym "??_R4failure@ios_base@std@@6B@"
	.addrsig_sym "??_R3failure@ios_base@std@@8"
	.addrsig_sym "??_R2failure@ios_base@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@failure@ios_base@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@system_error@std@@8"
	.addrsig_sym "??_R3system_error@std@@8"
	.addrsig_sym "??_R2system_error@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@_System_error@std@@8"
	.addrsig_sym "??_R3_System_error@std@@8"
	.addrsig_sym "??_R2_System_error@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@runtime_error@std@@8"
	.addrsig_sym "??_R3runtime_error@std@@8"
	.addrsig_sym "??_R2runtime_error@std@@8"
	.addrsig_sym "??_R4system_error@std@@6B@"
	.addrsig_sym "??_R4_System_error@std@@6B@"
	.addrsig_sym "??_R4runtime_error@std@@6B@"
	.addrsig_sym "?_Psave@?$_Facetptr@V?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@std@@2PEBVfacet@locale@2@EB"
	.addrsig_sym "??_R4?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@6B@"
	.addrsig_sym "??_R0?AV?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@@8"
	.addrsig_sym "??_R3?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.addrsig_sym "??_R2?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.addrsig_sym "??_R1A@?0A@EA@?$num_put@DV?$ostreambuf_iterator@DU?$char_traits@D@std@@@std@@@std@@8"
	.addrsig_sym "?_OptionsStorage@?1??__local_stdio_printf_options@@9@4_KA"
	.globl	_fltused
