/*
 * jdtrans.c
 *
 * Copyright (C) 1995-1997, Thomas G. Lane.
 * Modified 2000-2009 by Guido Vollbeding.
 * This file is part of the Independent JPEG Group's software.
 * For conditions of distribution and use, see the accompanying README file.
 *
 * This file contains library routines for transcoding decompression,
 * that is, reading raw DCT coefficient arrays from an input JPEG file.
 * The routines in jdapimin.c will also be needed by a transcoder.
 */

#define JPEG_INTERNALS
#include "jinclude.h"
#include "jpeglib.h"


/* Forward declarations */
LOCAL(void) transdecode_master_selection JPP((j_decompress_ptr cinfo));


/*
 * Read the coefficient arrays from a JPEG file.
 * jpeg_read_header must be completed before calling this.
 *
 * The entire image is read into a set of virtual coefficient-block arrays,
 * one per component.  The return value is a pointer to the array of
 * virtual-array descriptors.  These can be manipulated directly via the
 * JPEG memory manager, or handed off to jpeg_write_coefficients().
 * To release the memory occupied by the virtual arrays, call
 * jpeg_finish_decompress() when done with the data.
 *
 * An alternative usage is to simply obtain access to the coefficient arrays
 * during a buffered-image-mode decompression operation.  This is allowed
 * after any jpeg_finish_output() call.  The arrays can be accessed until
 * jpeg_finish_decompress() is called.  (Note that any call to the library
 * may reposition the arrays, so don't rely on access_virt_barray() results
 * to stay valid across library calls.)
 *
 * Returns NULL if suspended.  This case need be checked only if
 * a suspending data source is used.
 */



GLOBAL(jvirt_barray_ptr *)
jpeg_read_coefficients (j_decompress_ptr cinfo)
{
  if (cinfo->global_state == DSTATE_READY) {
    /* First call: initialize active modules */
    transdecode_master_selection(cinfo);
    cinfo->global_state = DSTATE_RDCOEFS;
  }
  if (cinfo->global_state == DSTATE_RDCOEFS) {
    /* Absorb whole file into the coef buffer */
    for (;;) {
      int retcode;
      /* Call progress monitor hook if present */
      if (cinfo->progress != NULL)
	(*cinfo->progress->progress_monitor) ((j_common_ptr) cinfo);
      /* Absorb some more input */
      //printf("jdtrans.c----------------------\n");
      retcode = (*cinfo->inputctl->consume_input) (cinfo);
      if (retcode == JPEG_SUSPENDED)
	return NULL;
      if (retcode == JPEG_REACHED_EOI)
	break;
      /* Advance progress counter if appropriate */
      if (cinfo->progress != NULL &&
	  (retcode == JPEG_ROW_COMPLETED || retcode == JPEG_REACHED_SOS)) {
	if (++cinfo->progress->pass_counter >= cinfo->progress->pass_limit) {
	  /* startup underestimated number of scans; ratchet up one scan */
	  cinfo->progress->pass_limit += (long) cinfo->total_iMCU_rows;
	}
      }
    }
    /* Set state so that jpeg_finish_decompress does the right thing */
    cinfo->global_state = DSTATE_STOPPING;
  }
  /* At this point we should be in state DSTATE_STOPPING if being used
   * standalone, or in state DSTATE_BUFIMAGE if being invoked to get access
   * to the coefficients during a full buffered-image-mode decompression.
   */
  if ((cinfo->global_state == DSTATE_STOPPING ||
       cinfo->global_state == DSTATE_BUFIMAGE) && cinfo->buffered_image) {
    return cinfo->coef->coef_arrays;
  }
  /* Oops, improper usage */
  ERREXIT1(cinfo, JERR_BAD_STATE, cinfo->global_state);
  return NULL;			/* keep compiler happy */
}


GLOBAL(void )
jpeg_read_coefficients_scan (j_decompress_ptr cinfo)
{
 
    unsigned int totalMcuNum = cinfo->MCU_rows_in_scan * cinfo->MCUs_per_row;
    //printf("totalMcuNum: %d\n",totalMcuNum);
    cinfo->num_mcu=0;
    for (int i = 0; i < totalMcuNum; ++i)//bian li mei yi ge mcu
    {
        if (FALSE == (*cinfo->entropy->decode_mcu_scan)(cinfo))
        {
            break;
        }
        cinfo->num_mcu+=1;
    }
}

GLOBAL(void )
jpeg_read_coefficients_mult (j_decompress_ptr cinfo,short * inputBuffer)
{
    unsigned int totalMcuNum = cinfo->MCU_rows_in_scan * cinfo->MCUs_per_row;
    unsigned int totalBlock = cinfo->blocks_in_MCU * totalMcuNum;// yi ge MCU you duo shao ge block ,zong gong you duo shao ge mcu
    JBLOCK **mcu_buffer = (JBLOCK**)malloc(sizeof(JBLOCK*) * cinfo->blocks_in_MCU);
    JBLOCK *block_buffer = (JBLOCK*)inputBuffer;
    cinfo->unread_marker = 0;
    cinfo->now_b1_id=1;
    cinfo->now_b2_id=cinfo->huff_scan.buffer_id[17000]-1;//di er ge xian cheng kai shi de buffer id;
    cinfo->now_b3_id=cinfo->huff_scan.buffer_id[24496]-1;
    cinfo->now_b4_id=cinfo->huff_scan.buffer_id[31900]-1;
    (*cinfo->src->retrun_file_start) (cinfo);

    for (int i = 0; i < totalMcuNum; ++i) {
        cinfo->line[1] = i;
        for (int j = 0; j < cinfo->blocks_in_MCU; ++j) {
            mcu_buffer[j] = block_buffer + cinfo->blocks_in_MCU * i + j;
        }
        (cinfo->entropy->decode_mcu_mult)(cinfo, mcu_buffer, 1);
    }
}

/*
 * Master selection of decompression modules for transcoding.
 * This substitutes for jdmaster.c's initialization of the full decompressor.
 */

LOCAL(void)
transdecode_master_selection (j_decompress_ptr cinfo)
{
  /* This is effectively a buffered-image operation. */
  cinfo->buffered_image = TRUE;

  /* Compute output image dimensions and related values. */
  jpeg_core_output_dimensions(cinfo);

  /* Entropy decoding: either Huffman or arithmetic coding. */
  if (cinfo->arith_code)
    jinit_arith_decoder(cinfo);
  else {
    jinit_huff_decoder(cinfo);
  }

  /* Always get a full-image coefficient buffer. */
  jinit_d_coef_controller(cinfo, TRUE);

  /* We can now tell the memory manager to allocate virtual arrays. */
  (*cinfo->mem->realize_virt_arrays) ((j_common_ptr) cinfo);

  /* Initialize input side of decompressor to consume first scan. */
  (*cinfo->inputctl->start_input_pass) (cinfo);

  /* Initialize progress monitoring. */
  if (cinfo->progress != NULL) {
    int nscans;
    /* Estimate number of scans to set pass_limit. */
    if (cinfo->progressive_mode) {
      /* Arbitrarily estimate 2 interleaved DC scans + 3 AC scans/component. */
      nscans = 2 + 3 * cinfo->num_components;
    } else if (cinfo->inputctl->has_multiple_scans) {
      /* For a nonprogressive multiscan file, estimate 1 scan per component. */
      nscans = cinfo->num_components;
    } else {
      nscans = 1;
    }
    cinfo->progress->pass_counter = 0L;
    cinfo->progress->pass_limit = (long) cinfo->total_iMCU_rows * nscans;
    cinfo->progress->completed_passes = 0;
    cinfo->progress->total_passes = 1;
  }
}
